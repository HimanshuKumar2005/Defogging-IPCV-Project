"""Tiny end-to-end tests for batch processing and manifest evaluation."""

from __future__ import annotations

import csv

import cv2
import numpy as np

from scripts.batch_process import process_directory, write_status_csv
from scripts.evaluate import evaluate_manifest, write_comparison_csv


def _write_rgb(path, color: tuple[int, int, int]) -> None:
    rgb = np.full((12, 12, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def test_batch_process_uses_method_specific_outputs(tmp_path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_rgb(input_dir / "scene.png", (90, 110, 130))
    rows = process_directory(input_dir, tmp_path / "batch", method="dcp")
    assert len(rows) == 1
    assert rows[0]["status"] == "success"
    assert rows[0]["method"] == "dcp"
    assert (tmp_path / "batch" / "dcp" / "scene_dcp.png").is_file()
    status_path = tmp_path / "batch" / "status.csv"
    write_status_csv(rows, status_path)
    with status_path.open(newline="", encoding="utf-8") as file:
        assert next(csv.DictReader(file))["status"] == "success"


def test_evaluation_manifest_processes_same_input_for_both_methods(tmp_path) -> None:
    hazy = tmp_path / "hazy.png"
    clear = tmp_path / "clear.png"
    _write_rgb(hazy, (90, 110, 130))
    _write_rgb(clear, (100, 120, 140))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", "image_path", "reference_path"])
        writer.writeheader()
        writer.writerow({"image_id": "scene", "image_path": hazy.name, "reference_path": clear.name})

    rows, summaries = evaluate_manifest(manifest, tmp_path / "evaluation", method="dcp")
    assert len(rows) == 1
    assert rows[0]["image_id"] == "scene"
    assert rows[0]["status"] == "success"
    assert rows[0]["method"] == "dcp"
    assert rows[0]["psnr"] != ""
    assert summaries[0]["method"] == "dcp"
    comparison = tmp_path / "comparison.csv"
    write_comparison_csv(rows, comparison)
    with comparison.open(newline="", encoding="utf-8") as file:
        exported = next(csv.DictReader(file))
    assert {"status", "psnr", "ssim", "processing_time_seconds"} <= set(exported)


def test_evaluation_marks_missing_reference_without_metrics(tmp_path) -> None:
    hazy = tmp_path / "hazy.png"
    _write_rgb(hazy, (90, 110, 130))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "reference_path"])
        writer.writeheader()
        writer.writerow({"image_path": hazy.name, "reference_path": ""})
    rows, _ = evaluate_manifest(manifest, tmp_path / "evaluation", method="dcp")
    assert rows[0]["status"] == "reference_invalid"
    assert rows[0]["psnr"] == ""
    assert rows[0]["ssim"] == ""
