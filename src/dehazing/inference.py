"""Inference orchestration for classical and pretrained dehazing methods."""

from __future__ import annotations

from typing import Any


def select_device(preferred: str = "auto") -> str:
    """Select CUDA when available, while retaining CPU support."""
    if preferred == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if preferred in {"auto", "cuda"} and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_inference(image: Any, method: str = "dcp", device: str = "auto") -> Any:
    """Dispatch an image to a dehazing method in a later implementation."""
    del image, device
    if method not in {"dcp", "aodnet"}:
        raise ValueError(f"Unsupported dehazing method: {method}")
    raise NotImplementedError("Inference orchestration is not implemented yet")
