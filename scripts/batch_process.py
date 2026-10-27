"""Batch-process a directory of RGB images with DCP and/or AOD-Net."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from dehazing.inference import dehaze_with_aodnet, dehaze_with_dcp
from dehazing.utils.image_io import load_image, save_image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _output_name(source: Path, method: str) -> str:
    return f"{source.stem}_{method}.png"


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    method: str = "both",
    checkpoint_path: str | Path | None = None,
    config: str | Path | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Process supported files and return one status row per method attempt."""
    source_dir = Path(input_dir)
    destination = Path(output_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {source_dir}")
    if method not in {"dcp", "aodnet", "both"}:
        raise ValueError("method must be dcp, aodnet, or both")

    methods = ["dcp", "aodnet"] if method == "both" else [method]
    rows: list[dict[str, Any]] = []
    for source in sorted(source_dir.iterdir()):
        if not source.is_file() or source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            image = load_image(source)
        except (FileNotFoundError, ValueError, OSError) as exc:
            for selected_method in methods:
                rows.append(
                    {
                        "image_id": source.stem,
                        "input_path": str(source),
                        "method": selected_method,
                        "status": "invalid_input",
                        "processing_time_seconds": "",
                        "output_path": "",
                        "error": str(exc),
                    }
                )
            continue

        for selected_method in methods:
            output_path = destination / selected_method / _output_name(source, selected_method)
            row: dict[str, Any] = {
                "image_id": source.stem,
                "input_path": str(source),
                "method": selected_method,
                "status": "failed",
                "processing_time_seconds": "",
                "output_path": str(output_path),
                "error": "",
            }
            try:
                if selected_method == "dcp":
                    result = dehaze_with_dcp(image, config)
                else:
                    result = dehaze_with_aodnet(image, checkpoint_path, device)
                save_image(result.image, output_path)
                row["status"] = "success"
                row["processing_time_seconds"] = f"{result.processing_time:.9f}"
            except FileNotFoundError as exc:
                row["status"] = "model_unavailable" if selected_method == "aodnet" else "failed"
                row["error"] = str(exc)
            except (RuntimeError, TypeError, ValueError, OSError) as exc:
                row["error"] = str(exc)
            rows.append(row)
    return rows


def write_status_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write batch status rows to CSV."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_id",
        "input_path",
        "method",
        "status",
        "processing_time_seconds",
        "output_path",
        "error",
    ]
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch dehaze an image directory.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/batch"))
    parser.add_argument("--method", choices=("dcp", "aodnet", "both"), default="both")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--status-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = process_directory(
        args.input_dir,
        args.output_dir,
        args.method,
        args.checkpoint,
        args.config,
        args.device,
    )
    status_path = args.status_csv or args.output_dir / "batch_status.csv"
    write_status_csv(rows, status_path)
    print(f"Processed {len(rows)} method attempts. Status CSV: {status_path}")


if __name__ == "__main__":
    main()
