"""Streamlit application for comparing DCP and pretrained AOD-Net dehazing."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.dehazing.inference import (
    DehazingResult,
    compare_dehazing_methods,
    dehaze_with_aodnet,
    dehaze_with_dcp,
    inspect_dcp,
    select_device,
)
from src.dehazing.metrics import calculate_psnr, calculate_ssim


def _uploaded_to_rgb(uploaded_file: Any) -> np.ndarray:
    """Decode an uploaded image as an RGB uint8 array."""
    try:
        with Image.open(uploaded_file) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid JPG, JPEG, or PNG image.") from exc


def _image_bytes(image: np.ndarray) -> bytes:
    """Encode an RGB float or uint8 image as PNG bytes for download."""
    values = np.asarray(image)
    if values.dtype != np.uint8:
        values = np.clip(values * 255.0, 0, 255).round().astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(values, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _dcp_config(st: Any) -> dict[str, dict[str, float | int]]:
    with st.sidebar.expander("DCP parameters", expanded=True):
        patch_size = st.slider("Patch size", 1, 31, 15, 2)
        omega = st.slider("Omega", 0.0, 1.0, 0.95, 0.01)
        t_min = st.slider("Minimum transmission", 0.01, 0.5, 0.1, 0.01)
        atmospheric_percentile = st.number_input(
            "Atmospheric-light percentile", 0.0001, 1.0, 0.001, format="%.4f"
        )
        guided_radius = st.slider("Guided-filter radius", 1, 100, 40)
        guided_epsilon = st.number_input(
            "Guided-filter epsilon", 0.000001, 1.0, 0.001, format="%.6f"
        )
    return {"dcp": {
        "patch_size": patch_size,
        "omega": omega,
        "t_min": t_min,
        "atmospheric_percentile": atmospheric_percentile,
        "guided_radius": guided_radius,
        "guided_epsilon": guided_epsilon,
    }}


def _metrics(reference: np.ndarray | None, result: DehazingResult) -> tuple[str, str]:
    if reference is None:
        return "Unavailable", "Unavailable"
    try:
        psnr = calculate_psnr(reference, result.image)
        ssim = calculate_ssim(reference, result.image)
    except (TypeError, ValueError):
        return "Unavailable", "Unavailable"
    return ("∞ dB" if np.isinf(psnr) else f"{psnr:.2f} dB", f"{ssim:.4f}")


def _comparison_table(st: Any, results: list[DehazingResult], reference: np.ndarray | None) -> None:
    rows = []
    for result in results:
        psnr, ssim = _metrics(reference, result)
        rows.append({
            "Method": result.method.upper(),
            "Inference time (s)": f"{result.processing_time:.4f}",
            "PSNR": psnr,
            "SSIM": ssim,
        })
    st.subheader("Compact comparison")
    st.table(rows)
    if reference is None:
        st.caption("PSNR and SSIM are unavailable without a valid aligned clear reference.")
    st.info("A visually pleasing result does not automatically have the best PSNR or SSIM.")


def _difference_view(st: Any, result: DehazingResult, reference: np.ndarray) -> None:
    try:
        reference_values = reference.astype(np.float32)
        if np.issubdtype(reference.dtype, np.integer):
            reference_values /= 255.0
        difference = np.abs(reference_values - result.image)
        difference = np.clip(difference * 4.0, 0.0, 1.0)
    except (ValueError, TypeError):
        return
    st.image(
        difference,
        caption=f"Absolute difference from reference ({result.method.upper()}); contrast enhanced 4×",
        use_container_width=True,
    )


def _result_download(st: Any, result: DehazingResult) -> None:
    st.download_button(
        f"Download {result.method.upper()} PNG",
        data=_image_bytes(result.image),
        file_name=f"{result.method}_dehazed.png",
        mime="image/png",
        key=f"download-{result.method}",
    )


def _display_results(
    st: Any,
    original: np.ndarray,
    results: list[DehazingResult],
    reference: np.ndarray | None,
    zoom: bool,
) -> None:
    """Render actual outputs in a three-panel or before/after layout."""
    panels = [("Original", original)] + [
        (result.method.upper(), result.image) for result in results
    ]
    columns = st.columns(len(panels))
    for column, (caption, image) in zip(columns, panels):
        column.image(image, caption=caption, use_container_width=True)
    if zoom:
        with st.expander("Zoomed detail inspection", expanded=True):
            height, width = original.shape[:2]
            crop_size = min(height, width, max(32, min(height, width) // 2))
            y0 = (height - crop_size) // 2
            x0 = (width - crop_size) // 2
            crop_columns = st.columns(len(panels))
            for column, (caption, image) in zip(crop_columns, panels):
                column.image(
                    image[y0:y0 + crop_size, x0:x0 + crop_size],
                    caption=f"{caption} center crop",
                    use_container_width=True,
                )
    _comparison_table(st, results, reference)
    for result in results:
        _result_download(st, result)
    if reference is not None:
        with st.expander("Visual difference from clear reference"):
            for result in results:
                _difference_view(st, result, reference)


def _display_dcp_diagnostics(st: Any, image: np.ndarray, config: dict[str, Any]) -> None:
    with st.expander("DCP diagnostics", expanded=False):
        show_dark = st.checkbox("Show dark channel", value=True)
        show_transmission = st.checkbox("Show transmission map", value=True)
        details = inspect_dcp(image, config)
        diagnostic_columns = st.columns(2)
        if show_dark:
            diagnostic_columns[0].image(
                details["dark_channel"],
                caption="Dark channel (darker pixels indicate stronger haze prior)",
                clamp=True,
                use_container_width=True,
            )
        if show_transmission:
            diagnostic_columns[1].image(
                details["transmission"],
                caption="Refined transmission map (brighter means more scene visibility)",
                clamp=True,
                use_container_width=True,
            )
        st.caption(
            "Diagnostic maps are generated by the same DCP execution family; "
            "they are not benchmark or fabricated outputs."
        )


def _run_batch(st: Any, files: list[Any], config: dict[str, Any], method: str,
               checkpoint: Path | None, device: str) -> None:
    rows = []
    for uploaded in files:
        try:
            image = _uploaded_to_rgb(uploaded)
            results = []
            if method in {"Classical DCP", "Compare Both"}:
                results.append(dehaze_with_dcp(image, config))
            if method in {"Pretrained AOD-Net", "Compare Both"}:
                results.append(dehaze_with_aodnet(image, checkpoint, device))
            for result in results:
                rows.append({
                    "Image": uploaded.name,
                    "Method": result.method.upper(),
                    "Status": "success",
                    "Time (s)": round(result.processing_time, 4),
                })
        except FileNotFoundError as exc:
            rows.append({"Image": uploaded.name, "Method": "AODNET", "Status": "model_unavailable", "Time (s)": None, "Error": str(exc)})
        except (RuntimeError, TypeError, ValueError, OSError) as exc:
            rows.append({"Image": uploaded.name, "Method": method, "Status": "failed", "Time (s)": None, "Error": str(exc)})
    st.subheader("Batch-processing summary")
    st.dataframe(rows, use_container_width=True)
    st.caption("Batch summaries contain only actual executions; unavailable methods are not replaced with fabricated results.")


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Streamlit is required. Install requirements.txt.") from exc

    st.set_page_config(page_title="Image Dehazing: Classical vs Deep Learning", page_icon="🌫️", layout="wide")
    st.title("Image Dehazing: Classical vs Deep Learning")
    st.write("Compare actual DCP and optional pretrained AOD-Net executions. No training is performed.")
    with st.sidebar:
        st.header("Controls")
        method = st.radio("Dehazing method", ("Classical DCP", "Pretrained AOD-Net", "Compare Both"))
        device = st.selectbox("Execution device", ("auto", "cpu", "cuda"))
        checkpoint_text = st.text_input("AOD-Net checkpoint path", "checkpoints/aodnet.pth")
        zoom = st.checkbox("Show zoomed details", value=False)
        batch_mode = st.checkbox("Batch upload mode", value=False)
        try:
            st.caption(f"Selected device: `{select_device(device)}`")
        except RuntimeError as exc:
            st.warning(str(exc))
    config = _dcp_config(st)
    checkpoint = Path(checkpoint_text).expanduser() if checkpoint_text.strip() else None
    uploaded = st.file_uploader(
        "Upload hazy image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=batch_mode,
    )
    reference_file = st.file_uploader("Optional clear-reference image", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Upload a JPG, JPEG, or PNG image to begin.")
        return
    if batch_mode:
        _run_batch(st, uploaded if isinstance(uploaded, list) else [uploaded], config, method, checkpoint, device)
        return
    try:
        image = _uploaded_to_rgb(uploaded)
        reference = _uploaded_to_rgb(reference_file) if reference_file else None
    except ValueError as exc:
        st.error(str(exc))
        return
    st.subheader("Before")
    st.image(image, caption=f"Original hazy RGB image ({image.shape[1]} × {image.shape[0]})", use_container_width=True)
    if st.button("Run dehazing", type="primary"):
        results: list[DehazingResult] = []
        if method == "Classical DCP":
            try:
                results = [dehaze_with_dcp(image, config)]
            except (TypeError, ValueError, OSError) as exc:
                st.error(f"DCP inference failed: {exc}")
        elif method == "Pretrained AOD-Net":
            try:
                results = [dehaze_with_aodnet(image, checkpoint, device)]
            except FileNotFoundError as exc:
                st.warning(f"AOD-Net is unavailable: {exc}")
                st.info("Install a compatible checkpoint or select Classical DCP.")
            except (RuntimeError, TypeError, ValueError) as exc:
                st.error(f"AOD-Net inference failed: {exc}")
        else:
            comparison = compare_dehazing_methods(image, checkpoint, config, device)
            if comparison["dcp"] is not None:
                results.append(comparison["dcp"])
            if comparison["aodnet"] is not None:
                results.append(comparison["aodnet"])
            elif "errors" in comparison:
                error = comparison["errors"]["aodnet"]
                st.warning(f"AOD-Net is unavailable or failed: {error}")
        if results:
            _display_results(st, image, results, reference, zoom)
            if any(result.method == "dcp" for result in results):
                _display_dcp_diagnostics(st, image, config)


if __name__ == "__main__":
    main()
