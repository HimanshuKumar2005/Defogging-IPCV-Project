"""Utilities for loading an optional pretrained AOD-Net checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_pretrained_model(checkpoint_path: str | Path, device: str = "cpu") -> Any:
    """Load a compatible AOD-Net checkpoint in a future implementation."""
    del device
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"AOD-Net checkpoint was not found: {path}")
    raise NotImplementedError("Pretrained AOD-Net loading is not implemented yet")
