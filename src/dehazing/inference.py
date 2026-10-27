"""Central inference interface for DCP and pretrained AOD-Net dehazing.

All public image functions accept RGB NumPy arrays as ``uint8`` values in
``[0, 255]`` or floating-point values in ``[0, 1]``. Results contain RGB
``float32`` images in ``[0, 1]``. AOD-Net inference never runs without a
successfully loaded compatible checkpoint.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .classical.dcp import dehaze_dcp, dehaze_dcp_details
from .metrics import measure_torch_inference
from .models.aodnet import AODNet
from .models.pretrained_loader import load_pretrained_model, predict

_MODEL_CACHE: dict[tuple[str, str], AODNet] = {}
_MODEL_CACHE_MAX = 8  # prevent unbounded memory growth


@dataclass(frozen=True)
class DehazingResult:
    """Output and measurements from one dehazing method."""

    image: np.ndarray
    method: str
    processing_time: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3) in RGB order")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image dimensions must be non-zero")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    if np.issubdtype(image.dtype, np.integer):
        if image.dtype != np.uint8:
            raise ValueError("integer images must use uint8 values in [0, 255]")
        normalized = image.astype(np.float32) / 255.0
    else:
        normalized = image.astype(np.float32, copy=False)
        if np.min(normalized) < 0 or np.max(normalized) > 1:
            raise ValueError("floating-point images must contain values in [0, 1]")
    return np.clip(normalized, 0.0, 1.0)


def select_device(preferred: str | None = "auto") -> str:
    """Select CUDA when available, otherwise CPU for automatic selection."""
    requested = "auto" if preferred is None else preferred.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or None")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise RuntimeError("PyTorch is required for CUDA inference")
        return "cpu"
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but is not available")
    return "cpu"


def dehaze_with_dcp(
    image: np.ndarray, config: Mapping[str, Any] | str | Path | None = None
) -> DehazingResult:
    """Run DCP and return its image, timing, and algorithm metadata."""
    normalized = _validate_image(image)
    started = time.perf_counter()
    output = dehaze_dcp(normalized, config)
    elapsed = time.perf_counter() - started
    return DehazingResult(
        image=np.asarray(output, dtype=np.float32),
        method="dcp",
        processing_time=elapsed,
        metadata={"input_shape": tuple(image.shape), "output_range": "[0, 1]"},
    )


def inspect_dcp(
    image: np.ndarray, config: Mapping[str, Any] | str | Path | None = None
) -> dict[str, np.ndarray | float]:
    """Return DCP output and diagnostic maps with one measured execution."""
    normalized = _validate_image(image)
    started = time.perf_counter()
    details = dehaze_dcp_details(normalized, config)
    details["processing_time"] = time.perf_counter() - started
    return details


def _get_aodnet_model(checkpoint_path: str | Path, device: str) -> AODNet:
    path = Path(checkpoint_path).expanduser().resolve()
    key = (str(path), device)
    if key not in _MODEL_CACHE:
        if len(_MODEL_CACHE) >= _MODEL_CACHE_MAX:
            # Evict the oldest entry to cap memory usage.
            _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
        _MODEL_CACHE[key] = load_pretrained_model(path, device)
    return _MODEL_CACHE[key]


def dehaze_with_aodnet(
    image: np.ndarray,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> DehazingResult:
    """Run AOD-Net using a verified checkpoint, without resizing the image."""
    normalized = _validate_image(image)
    selected_device = select_device(device)
    checkpoint = checkpoint_path or Path("checkpoints") / "aodnet.pth"
    model = _get_aodnet_model(checkpoint, selected_device)

    import torch

    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
    tensor = tensor.to(selected_device)
    output_tensor, elapsed = measure_torch_inference(
        lambda: predict(model, tensor), selected_device
    )
    output = output_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    output = np.nan_to_num(np.clip(output, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
    return DehazingResult(
        image=output.astype(np.float32),
        method="aodnet",
        processing_time=elapsed,
        metadata={
            "input_shape": tuple(image.shape),
            "output_range": "[0, 1]",
            "device": selected_device,
            "checkpoint": str(Path(checkpoint)),
            "resized": False,
        },
    )


def compare_dehazing_methods(
    image: np.ndarray,
    checkpoint_path: str | Path | None = None,
    config: Mapping[str, Any] | str | Path | None = None,
    device: str | None = None,
) -> dict[str, DehazingResult | None]:
    """Run both methods, preserving DCP output if optional AOD-Net is absent.

    The returned mapping always contains ``dcp``. ``aodnet`` is ``None`` when
    its checkpoint is missing or invalid, with the failure message in
    ``errors['aodnet']``.
    """
    results: dict[str, Any] = {
        "dcp": dehaze_with_dcp(image, config),
        "aodnet": None,
    }
    try:
        results["aodnet"] = dehaze_with_aodnet(image, checkpoint_path, device)
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
        results["errors"] = {"aodnet": str(exc)}
    return results


def run_inference(
    image: np.ndarray,
    method: str = "dcp",
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    config: Mapping[str, Any] | str | Path | None = None,
) -> DehazingResult:
    """Dispatch one image to either DCP or checkpoint-backed AOD-Net."""
    normalized_method = method.lower()
    if normalized_method == "dcp":
        return dehaze_with_dcp(image, config)
    if normalized_method == "aodnet":
        return dehaze_with_aodnet(image, checkpoint_path, device)
    raise ValueError(f"Unsupported dehazing method: {method}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dehaze one RGB image.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--method", choices=("dcp", "aodnet"), default="dcp")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for single-image inference."""
    args = _parse_args()
    input_path = args.image_path.resolve()
    output_path = args.output_path.resolve()
    if input_path == output_path:
        raise ValueError("output path must differ from the input image path")
    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read input image: {input_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = run_inference(
        image_rgb,
        args.method,
        args.device,
        args.checkpoint,
        args.config,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(
        str(output_path),
        cv2.cvtColor((result.image * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR),
    )
    if not success:
        raise OSError(f"Unable to write output image: {output_path}")
    print(f"{result.method}: {result.processing_time:.4f}s -> {output_path}")


if __name__ == "__main__":
    main()
