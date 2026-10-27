"""RGB image loading and saving helpers."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as an RGB uint8 NumPy array."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Save an RGB uint8 or float image, converting explicitly to BGR."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    values = image.astype(np.float32, copy=False)
    if np.issubdtype(image.dtype, np.integer):
        values = values / 255.0
    if not np.isfinite(values).all() or np.min(values) < 0 or np.max(values) > 1:
        raise ValueError("image values must be finite and in [0, 1] or uint8 [0, 255]")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(
        str(destination),
        cv2.cvtColor((values * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR),
    )
    if not success:
        raise OSError(f"Unable to write image: {destination}")
