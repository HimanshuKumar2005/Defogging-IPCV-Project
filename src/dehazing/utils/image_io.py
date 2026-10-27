"""Image loading and saving helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_image(path: str | Path) -> Any:
    """Load an image in a future implementation."""
    raise NotImplementedError(f"Image loading is not implemented yet: {path}")


def save_image(image: Any, path: str | Path) -> None:
    """Save an image in a future implementation."""
    del image
    raise NotImplementedError(f"Image saving is not implemented yet: {path}")
