"""Tests for image quality metrics and evaluation records."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from dehazing.metrics import (
    aggregate_records,
    calculate_psnr,
    calculate_ssim,
    create_record,
    evaluate_pair,
    export_records_csv,
    measure_inference,
)


def _reference() -> np.ndarray:
    return np.full((16, 16, 3), 0.5, dtype=np.float32)


def test_identical_images_have_expected_metrics() -> None:
    image = _reference()
    assert math.isinf(calculate_psnr(image, image))
    assert calculate_ssim(image, image) == pytest.approx(1.0)


def test_known_difference_uses_normalized_data_range() -> None:
    reference = np.zeros((8, 8, 3), dtype=np.float32)
    prediction = np.ones_like(reference)
    assert calculate_psnr(reference, prediction) == pytest.approx(0.0)
    assert calculate_ssim(reference, prediction) < 0.001


def test_uint8_and_float_inputs_are_consistent() -> None:
    reference = np.zeros((8, 8, 3), dtype=np.uint8)
    prediction = np.full_like(reference, 128)
    assert calculate_psnr(reference, prediction) == pytest.approx(
        calculate_psnr(reference.astype(np.float32), prediction.astype(np.float32) / 255),
        abs=1e-5,
    )


def test_unavailable_reference_metrics_are_explicit() -> None:
    result = evaluate_pair(None, _reference())
    assert result == {"psnr": None, "ssim": None}
    record = create_record("image-1", "dcp", _reference())
    assert record.psnr is None
    assert record.ssim is None


def test_invalid_dimensions_and_small_ssim_images_raise() -> None:
    with pytest.raises(ValueError):
        calculate_psnr(np.zeros((4, 4, 3)), np.zeros((4, 5, 3)))
    with pytest.raises(ValueError):
        calculate_ssim(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)))


def test_timing_and_aggregation() -> None:
    value, elapsed = measure_inference(lambda: 42)
    assert value == 42
    assert elapsed >= 0
    records = [
        create_record("a", "dcp", _reference(), _reference(), 0.2),
        create_record("b", "dcp", np.zeros_like(_reference()), _reference(), 0.4),
        create_record("c", "dcp", _reference(), inference_time_seconds=0.6),
    ]
    summary = aggregate_records(records)
    assert summary["psnr"]["count"] == 1
    assert summary["ssim"]["count"] == 2
    assert summary["inference_time_seconds"]["mean"] == pytest.approx(0.4)
    assert summary["inference_time_seconds"]["std"] == pytest.approx(
        np.std([0.2, 0.4, 0.6])
    )


def test_csv_export_preserves_per_image_rows(tmp_path) -> None:
    records = [create_record("a", "dcp", _reference(), _reference(), 0.1)]
    output = tmp_path / "reports" / "metrics.csv"
    export_records_csv(records, output)
    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["image_id"] == "a"
    assert rows[0]["psnr"] == "inf"
