"""Dark Channel Prior (DCP) single-image dehazing.

Public functions accept RGB images as ``numpy.ndarray`` values. Floating-point
images may use the range ``[0, 1]``; ``uint8`` images in ``[0, 255]`` are also
accepted and normalized internally. Results are returned as ``float32`` RGB
arrays in ``[0, 1]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

DEFAULT_CONFIG: dict[str, float | int] = {
    "patch_size": 15,
    "omega": 0.95,
    "t_min": 0.1,
    "atmospheric_percentile": 0.001,
    "guided_radius": 40,
    "guided_epsilon": 1e-3,
}


def _validate_rgb_image(image: np.ndarray) -> np.ndarray:
    """Validate an RGB image and convert it to finite float32 values in [0, 1]."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3) in RGB order")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image dimensions must be non-zero")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")

    image_float = image.astype(np.float32, copy=False)
    if np.issubdtype(image.dtype, np.integer):
        if image.dtype != np.uint8:
            raise ValueError("integer images must use uint8 values in [0, 255]")
        image_float = image_float / 255.0
    elif np.min(image_float) < 0.0 or np.max(image_float) > 1.0:
        raise ValueError("floating-point images must contain values in [0, 1]")
    return np.clip(image_float, 0.0, 1.0)


def _odd_window_size(window_size: int) -> int:
    if not isinstance(window_size, (int, np.integer)) or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    return int(window_size) if window_size % 2 else int(window_size) + 1


def get_dark_channel(image: np.ndarray, window_size: int) -> np.ndarray:
    """Compute the local minimum dark channel for an RGB image."""
    normalized = _validate_rgb_image(image)
    size = _odd_window_size(window_size)
    channel_minimum = np.min(normalized, axis=2)
    return cv2.erode(
        channel_minimum,
        np.ones((size, size), dtype=np.uint8),
        borderType=cv2.BORDER_REPLICATE,
    ).astype(np.float32)


def estimate_atmospheric_light(
    image: np.ndarray, dark_channel: np.ndarray
) -> np.ndarray:
    """Estimate atmospheric light from the brightest dark-channel candidates."""
    normalized = _validate_rgb_image(image)
    dark = np.asarray(dark_channel, dtype=np.float32)
    if dark.shape != normalized.shape[:2]:
        raise ValueError("dark_channel must match the image height and width")
    if not np.isfinite(dark).all():
        raise ValueError("dark_channel must contain only finite values")

    flat_dark = dark.reshape(-1)
    flat_image = normalized.reshape(-1, 3)
    candidate_count = max(1, int(np.ceil(flat_dark.size * 0.001)))
    candidate_indices = np.argpartition(flat_dark, -candidate_count)[-candidate_count:]
    brightness = np.sum(flat_image[candidate_indices], axis=1)
    atmospheric = flat_image[candidate_indices[np.argmax(brightness)]]
    return np.clip(atmospheric, 1e-6, 1.0).astype(np.float32)


def estimate_transmission(
    image: np.ndarray,
    atmospheric_light: np.ndarray,
    omega: float,
    window_size: int,
) -> np.ndarray:
    """Estimate the coarse transmission map using normalized image intensities."""
    normalized = _validate_rgb_image(image)
    atmospheric = np.asarray(atmospheric_light, dtype=np.float32)
    if atmospheric.shape != (3,) or not np.isfinite(atmospheric).all():
        raise ValueError("atmospheric_light must be a finite RGB vector")
    if np.any(atmospheric <= 0):
        raise ValueError("atmospheric_light values must be positive")
    if not np.isfinite(omega) or not 0.0 <= omega <= 1.0:
        raise ValueError("omega must be finite and in the range [0, 1]")

    normalized_image = normalized / np.maximum(atmospheric, 1e-6)
    normalized_image = np.clip(normalized_image, 0.0, 1.0)
    normalized_dark = get_dark_channel(normalized_image, window_size)
    return np.clip(1.0 - float(omega) * normalized_dark, 0.0, 1.0).astype(
        np.float32
    )


def _box_filter(values: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.boxFilter(
        values.astype(np.float32),
        ddepth=-1,
        ksize=(size, size),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )


def refine_transmission(
    image: np.ndarray, transmission: np.ndarray, radius: int, epsilon: float
) -> np.ndarray:
    """Refine transmission with ximgproc guided filtering or a box-filter fallback.

    If OpenCV was installed without ``opencv-contrib-python``, ``ximgproc`` is
    unavailable. In that case a vectorized grayscale guided-filter
    implementation is used, which preserves the same edge-aware behavior.
    """
    normalized = _validate_rgb_image(image)
    transmission_array = np.asarray(transmission, dtype=np.float32)
    if transmission_array.shape != normalized.shape[:2]:
        raise ValueError("transmission must match the image height and width")
    if not np.isfinite(transmission_array).all():
        raise ValueError("transmission must contain only finite values")
    if not isinstance(radius, (int, np.integer)) or radius < 1:
        raise ValueError("radius must be a positive integer")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")

    guide = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY).astype(np.float32)
    source = np.clip(transmission_array, 0.0, 1.0)
    ximgproc = getattr(cv2, "ximgproc", None)
    guided_filter = getattr(ximgproc, "guidedFilter", None)
    if guided_filter is not None:
        refined = guided_filter(guide, source, int(radius), float(epsilon))
        return np.clip(refined, 0.0, 1.0).astype(np.float32)

    mean_guide = _box_filter(guide, int(radius))
    mean_source = _box_filter(source, int(radius))
    covariance = _box_filter(guide * source, int(radius)) - mean_guide * mean_source
    variance = _box_filter(guide * guide, int(radius)) - mean_guide * mean_guide
    coefficient = covariance / (variance + float(epsilon))
    intercept = mean_source - coefficient * mean_guide
    refined = _box_filter(coefficient, int(radius)) * guide + _box_filter(
        intercept, int(radius)
    )
    return np.clip(refined, 0.0, 1.0).astype(np.float32)


def recover_scene_radiance(
    image: np.ndarray,
    transmission: np.ndarray,
    atmospheric_light: np.ndarray,
    t0: float,
) -> np.ndarray:
    """Recover scene radiance from the atmospheric scattering model."""
    normalized = _validate_rgb_image(image)
    transmission_array = np.asarray(transmission, dtype=np.float32)
    atmospheric = np.asarray(atmospheric_light, dtype=np.float32)
    if transmission_array.shape != normalized.shape[:2]:
        raise ValueError("transmission must match the image height and width")
    if atmospheric.shape != (3,) or not np.isfinite(atmospheric).all():
        raise ValueError("atmospheric_light must be a finite RGB vector")
    if not np.isfinite(transmission_array).all():
        raise ValueError("transmission must contain only finite values")
    if not np.isfinite(t0) or not 0.0 < t0 <= 1.0:
        raise ValueError("t0 must be finite and in the range (0, 1]")

    safe_transmission = np.maximum(transmission_array, float(t0))[..., None]
    recovered = (normalized - atmospheric) / safe_transmission + atmospheric
    return np.nan_to_num(
        np.clip(recovered, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0
    ).astype(np.float32)


def _load_config(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    values: dict[str, Any] = dict(DEFAULT_CONFIG)
    if config is None:
        return values
    if isinstance(config, Mapping):
        section = config.get("dcp", config)
        if not isinstance(section, Mapping):
            raise ValueError("config['dcp'] must be a mapping")
        values.update(section)
        return values
    if isinstance(config, (str, Path)):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load a YAML DCP config") from exc
        with Path(config).open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        return _load_config(loaded)
    raise TypeError("config must be a mapping, YAML path, or None")


def dehaze_dcp(
    image: np.ndarray, config: Mapping[str, Any] | str | Path | None = None
) -> np.ndarray:
    """Run the complete DCP dehazing pipeline and return an RGB float32 image."""
    settings = _load_config(config)
    patch_size = settings["patch_size"]
    omega = settings["omega"]
    t_min = settings["t_min"]
    percentile = settings["atmospheric_percentile"]
    if not np.isfinite(percentile) or not 0 < percentile <= 1:
        raise ValueError("atmospheric_percentile must be in the range (0, 1]")

    normalized = _validate_rgb_image(image)
    dark_channel = get_dark_channel(normalized, patch_size)
    atmospheric = estimate_atmospheric_light(normalized, dark_channel)
    # The atmospheric percentile is configurable; select it by replacing the
    # default candidate fraction for callers that need a different percentile.
    if percentile != 0.001:
        flat_dark = dark_channel.reshape(-1)
        count = max(1, int(np.ceil(flat_dark.size * percentile)))
        indices = np.argpartition(flat_dark, -count)[-count:]
        candidates = normalized.reshape(-1, 3)[indices]
        atmospheric = np.clip(candidates[np.argmax(np.sum(candidates, axis=1))], 1e-6, 1)
    transmission = estimate_transmission(normalized, atmospheric, omega, patch_size)
    refined = refine_transmission(
        normalized,
        transmission,
        settings["guided_radius"],
        settings["guided_epsilon"],
    )
    return recover_scene_radiance(normalized, refined, atmospheric, t_min)


def dehaze(image: np.ndarray) -> np.ndarray:
    """Backward-compatible shorthand for :func:`dehaze_dcp`."""
    return dehaze_dcp(image)
