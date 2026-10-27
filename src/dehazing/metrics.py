"""Image-quality metrics for evaluation reports."""

from __future__ import annotations

from typing import Any


def evaluate_pair(reference: Any, prediction: Any) -> dict[str, float]:
    """Calculate evaluation metrics in a future implementation."""
    del reference, prediction
    raise NotImplementedError("Evaluation metrics are not implemented yet")
