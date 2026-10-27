"""Safe loading of optional pretrained AOD-Net weights.

Only state-dict checkpoints are accepted. ``torch.load(..., weights_only=True)``
prevents arbitrary Python object deserialization. The loader never falls back
to random weights.
"""

from __future__ import annotations

import pickle  # used only as an exception type in the except clause below
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .aodnet import AODNet


class CheckpointError(RuntimeError):
    """Raised when an AOD-Net checkpoint is missing or incompatible."""


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint must contain a PyTorch state-dict mapping")
    candidate: Any = payload.get("state_dict", payload)
    if not isinstance(candidate, Mapping) or not candidate:
        raise CheckpointError("checkpoint does not contain a non-empty state_dict")
    if not all(isinstance(key, str) for key in candidate):
        raise CheckpointError("checkpoint state_dict keys must be strings")
    if not all(isinstance(value, torch.Tensor) for value in candidate.values()):
        raise CheckpointError("checkpoint state_dict values must be tensors")
    return candidate


def load_pretrained_model(
    checkpoint_path: str | Path, device: str | torch.device = "cpu"
) -> AODNet:
    """Load a verified AOD-Net state dict onto CPU or CUDA in eval mode.

    The referenced repositories do not currently publish a verifiable PyTorch
    checkpoint at a stable URL, so callers must provide a local checkpoint.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"AOD-Net checkpoint not found: {path}. "
            "Place a compatible state-dict checkpoint at this path."
        )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise CheckpointError("CUDA was requested but is not available")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = _extract_state_dict(payload)
        model = AODNet()
        model.load_state_dict(state_dict, strict=True)
    except (OSError, RuntimeError, TypeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise CheckpointError(
            f"Unable to load a compatible AOD-Net state dict from {path}: {exc}"
        ) from exc
    model.to(target_device)
    model.eval()
    return model


@torch.inference_mode()
def predict(model: AODNet, image: torch.Tensor) -> torch.Tensor:
    """Run inference only with a loaded, evaluation-mode AOD-Net."""
    if not isinstance(model, AODNet):
        raise TypeError("model must be an AODNet instance loaded by this module")
    if model.training:
        raise ValueError("model must be in evaluation mode")
    return model(image)
