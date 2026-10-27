"""Image-quality metrics and evaluation records for dehazing results.

Images use the same convention as the inference API: RGB ``uint8`` arrays in
``[0, 255]`` or floating-point arrays in ``[0, 1]``. PSNR and SSIM require a
real clear reference image; this module never substitutes an input, output, or
synthetic image as a reference. If a reference is unavailable, the metric is
represented by ``None``.

PSNR measures pixel-wise reconstruction error: higher is better, and identical
images have infinite PSNR. SSIM compares local luminance, contrast, and
structure: values near 1 indicate greater structural similarity. Neither
metric alone measures perceptual quality perfectly, and both depend on the
chosen reference and image alignment.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

T = TypeVar("T")
MetricValue = float | None


def _validate_image(image: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must have shape (height, width, 3) in RGB order")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{name} dimensions must be non-zero")
    if not np.isfinite(image).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.issubdtype(image.dtype, np.integer):
        if image.dtype != np.uint8:
            raise ValueError(f"{name} integer values must use uint8")
        normalized = image.astype(np.float32) / 255.0
    else:
        normalized = image.astype(np.float32, copy=False)
        if np.min(normalized) < 0 or np.max(normalized) > 1:
            raise ValueError(f"{name} floating-point values must be in [0, 1]")
    return np.clip(normalized, 0.0, 1.0)


def _validate_pair(reference: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference_normalized = _validate_image(reference, "reference")
    prediction_normalized = _validate_image(prediction, "prediction")
    if reference_normalized.shape != prediction_normalized.shape:
        raise ValueError("reference and prediction must have identical dimensions")
    return reference_normalized, prediction_normalized


def calculate_psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Return RGB PSNR in dB using ``data_range=1.0``.

    Identical images correctly return ``math.inf``.
    """
    reference_normalized, prediction_normalized = _validate_pair(reference, prediction)
    if np.array_equal(reference_normalized, prediction_normalized):
        return math.inf
    value = peak_signal_noise_ratio(
        reference_normalized, prediction_normalized, data_range=1.0
    )
    return float(value)


def calculate_ssim(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Return multichannel RGB SSIM using ``data_range=1.0``."""
    reference_normalized, prediction_normalized = _validate_pair(reference, prediction)
    height, width = reference_normalized.shape[:2]
    win_size = min(7, height, width)
    if win_size < 3:
        raise ValueError("SSIM requires images with height and width at least 3")
    if win_size % 2 == 0:
        win_size -= 1
    value = structural_similarity(
        reference_normalized,
        prediction_normalized,
        data_range=1.0,
        channel_axis=-1,
        win_size=win_size,
    )
    return float(value)


def evaluate_pair(
    reference: np.ndarray | None, prediction: np.ndarray
) -> dict[str, MetricValue]:
    """Return PSNR and SSIM, or ``None`` values when no reference is supplied."""
    if reference is None:
        _validate_image(prediction, "prediction")
        return {"psnr": None, "ssim": None}
    return {
        "psnr": calculate_psnr(reference, prediction),
        "ssim": calculate_ssim(reference, prediction),
    }


def measure_inference(
    inference_fn: Callable[[], T],
    *,
    synchronize: Callable[[], None] | None = None,
) -> tuple[T, float]:
    """Run a callable and measure steady-state inference time in seconds.

    Model construction and checkpoint loading must happen before ``inference_fn``
    is supplied. For CUDA callers, pass a synchronization callback such as
    ``torch.cuda.synchronize`` before and after the timed call.
    """
    if synchronize is not None:
        synchronize()
    started = time.perf_counter()
    result = inference_fn()
    if synchronize is not None:
        synchronize()
    return result, time.perf_counter() - started


def measure_torch_inference(
    inference_fn: Callable[[], T], device: str = "cpu"
) -> tuple[T, float]:
    """Measure inference and synchronize CUDA when the selected device is CUDA."""
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA timing requested but CUDA is unavailable")
        return measure_inference(inference_fn, synchronize=torch.cuda.synchronize)
    return measure_inference(inference_fn)


@dataclass(frozen=True)
class ImageMetricRecord:
    """Metrics and timing for one image/method pair."""

    image_id: str
    method: str
    psnr: MetricValue
    ssim: MetricValue
    inference_time_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def create_record(
    image_id: str,
    method: str,
    prediction: np.ndarray,
    reference: np.ndarray | None = None,
    inference_time_seconds: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ImageMetricRecord:
    """Create one per-image record without inventing reference metrics."""
    metrics = evaluate_pair(reference, prediction)
    if inference_time_seconds is not None and (
        not math.isfinite(inference_time_seconds) or inference_time_seconds < 0
    ):
        raise ValueError("inference_time_seconds must be finite and non-negative")
    return ImageMetricRecord(
        image_id=str(image_id),
        method=str(method),
        psnr=metrics["psnr"],
        ssim=metrics["ssim"],
        inference_time_seconds=inference_time_seconds,
        metadata=dict(metadata or {}),
    )


def aggregate_records(records: Iterable[ImageMetricRecord]) -> dict[str, dict[str, float | int | None]]:
    """Aggregate available numeric metrics using mean and population stddev."""
    materialized = list(records)
    if not materialized:
        raise ValueError("at least one record is required")
    aggregated: dict[str, dict[str, float | int | None]] = {}
    for metric in ("psnr", "ssim", "inference_time_seconds"):
        values = [
            float(getattr(record, metric))
            for record in materialized
            if getattr(record, metric) is not None
            and math.isfinite(float(getattr(record, metric)))
        ]
        aggregated[metric] = {
            "count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
        }
    return aggregated


def export_records_csv(
    records: Iterable[ImageMetricRecord], output_path: str | Path
) -> None:
    """Write per-image records to CSV, leaving unavailable values blank."""
    materialized = list(records)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "method",
        "psnr",
        "ssim",
        "inference_time_seconds",
        "metadata",
    ]
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in materialized:
            row = asdict(record)
            row["metadata"] = str(row["metadata"])
            writer.writerow(row)
