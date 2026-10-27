"""Manifest-driven evaluation for DCP and checkpoint-backed AOD-Net."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from dehazing.inference import dehaze_with_aodnet, dehaze_with_dcp
from dehazing.metrics import aggregate_records, create_record
from dehazing.utils.image_io import load_image, save_image


def read_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    """Read a CSV manifest with image_path, reference_path, and optional image_id."""
    path = Path(manifest_path)
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"image_path", "reference_path"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("manifest must contain image_path and reference_path columns")
        rows = []
        for index, row in enumerate(reader, start=1):
            image_path = Path(row["image_path"])
            reference_path = Path(row["reference_path"]) if row["reference_path"] else Path()
            rows.append(
                {
                    "image_id": row.get("image_id") or f"image_{index:04d}",
                    "image_path": str(image_path if image_path.is_absolute() else path.parent / image_path),
                    "reference_path": str(
                        reference_path
                        if reference_path.is_absolute()
                        else path.parent / reference_path
                    )
                    if row["reference_path"]
                    else "",
                }
            )
    return rows


def _reference_for_row(row: dict[str, str]) -> tuple[np.ndarray | None, str | None]:
    if not row["reference_path"]:
        return None, "reference_missing"
    try:
        return load_image(row["reference_path"]), None
    except (FileNotFoundError, ValueError, OSError) as exc:
        return None, f"reference_invalid: {exc}"


def evaluate_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    method: str = "both",
    checkpoint_path: str | Path | None = None,
    config: str | Path | None = None,
    device: str | None = None,
    generate_visuals: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate every manifest pair with the same input set for each method."""
    if method not in {"dcp", "aodnet", "both"}:
        raise ValueError("method must be dcp, aodnet, or both")
    methods = ["dcp", "aodnet"] if method == "both" else [method]
    rows: list[dict[str, Any]] = []
    records = {selected_method: [] for selected_method in methods}
    destination = Path(output_dir)

    for manifest_row in read_manifest(manifest_path):
        image_id = manifest_row["image_id"]
        try:
            image = load_image(manifest_row["image_path"])
        except (FileNotFoundError, ValueError, OSError) as exc:
            for selected_method in methods:
                rows.append(_error_row(manifest_row, selected_method, "invalid_input", str(exc)))
            continue
        reference, reference_error = _reference_for_row(manifest_row)
        if reference is not None and reference.shape != image.shape:
            reference = None
            reference_error = "reference_invalid: dimensions do not match hazy image"

        for selected_method in methods:
            row = _error_row(manifest_row, selected_method, "failed", "")
            try:
                result = (
                    dehaze_with_dcp(image, config)
                    if selected_method == "dcp"
                    else dehaze_with_aodnet(image, checkpoint_path, device)
                )
                output_path = destination / selected_method / f"{image_id}.png"
                if generate_visuals:
                    save_image(result.image, output_path)
                    row["output_path"] = str(output_path)
                record = create_record(
                    image_id,
                    selected_method,
                    result.image,
                    reference,
                    result.processing_time,
                    {"reference_status": reference_error or "valid"},
                )
                records[selected_method].append(record)
                row.update(
                    {
                        "status": "reference_invalid" if reference_error else "success",
                        "processing_time_seconds": f"{result.processing_time:.9f}",
                        "psnr": "" if record.psnr is None else record.psnr,
                        "ssim": "" if record.ssim is None else record.ssim,
                        "error": reference_error or "",
                    }
                )
            except FileNotFoundError as exc:
                row.update({"status": "model_unavailable" if selected_method == "aodnet" else "failed", "error": str(exc)})
            except (RuntimeError, TypeError, ValueError, OSError) as exc:
                row["error"] = str(exc)
            rows.append(row)

    summaries = [
        {"method": selected_method, **metrics}
        for selected_method, metrics in (
            (selected_method, aggregate_records(method_records))
            for selected_method, method_records in records.items()
            if method_records
        )
    ]
    return rows, summaries


def _error_row(manifest_row: dict[str, str], method: str, status: str, error: str) -> dict[str, Any]:
    return {
        "image_id": manifest_row["image_id"],
        "image_path": manifest_row["image_path"],
        "reference_path": manifest_row["reference_path"],
        "method": method,
        "status": status,
        "processing_time_seconds": "",
        "psnr": "",
        "ssim": "",
        "output_path": "",
        "error": error,
    }


def write_comparison_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_id", "image_path", "reference_path", "method", "status",
        "processing_time_seconds", "psnr", "ssim", "output_path", "error",
    ]
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(summaries: list[dict[str, Any]], output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "psnr", "ssim", "inference_time_seconds"]
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row = {"method": summary["method"]}
            for metric, values in summary.items():
                if metric != "method":
                    row[metric] = f"{values['mean']} +/- {values['std']} (n={values['count']})"
            writer.writerow(row)


def write_metric_plot(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Plot only actual finite PSNR/SSIM values from successful evaluations."""
    import matplotlib.pyplot as plt

    valid = [row for row in rows if row["psnr"] != "" and row["ssim"] != ""]
    if not valid:
        raise ValueError("no numeric PSNR/SSIM results are available for plotting")
    methods = sorted({row["method"] for row in valid})
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for selected_method in methods:
        method_rows = [row for row in valid if row["method"] == selected_method]
        axes[0].plot([float(row["psnr"]) for row in method_rows], label=selected_method)
        axes[1].plot([float(row["ssim"]) for row in method_rows], label=selected_method)
    axes[0].set_title("PSNR (dB)")
    axes[1].set_title("SSIM")
    for axis in axes:
        axis.set_xlabel("Evaluation image")
        axis.legend()
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dehazing methods from a CSV manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--method", choices=("dcp", "aodnet", "both"), default="both")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--visuals", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows, summaries = evaluate_manifest(
        args.manifest, args.output_dir, args.method, args.checkpoint,
        args.config, args.device, args.visuals,
    )
    comparison_path = args.output_dir / "comparison.csv"
    summary_path = args.output_dir / "summary.csv"
    write_comparison_csv(rows, comparison_path)
    write_summary_csv(summaries, summary_path)
    if args.plot:
        write_metric_plot(rows, args.output_dir / "metrics.png")
    print(f"Evaluation rows: {len(rows)}")
    print(f"Comparison CSV: {comparison_path}")
    print(f"Summary CSV: {summary_path}")
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
