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
    """Render DCP controls and return the selected configuration."""
    with st.sidebar.expander("DCP parameters", expanded=True):
        patch_size = st.slider("Patch size", min_value=1, max_value=31, value=15, step=2)
        omega = st.slider("Omega", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
        t_min = st.slider(
            "Minimum transmission", min_value=0.01, max_value=0.5, value=0.1, step=0.01
        )
        atmospheric_percentile = st.number_input(
            "Atmospheric-light percentile",
            min_value=0.0001,
            max_value=1.0,
            value=0.001,
            format="%.4f",
        )
        guided_radius = st.slider(
            "Guided-filter radius", min_value=1, max_value=100, value=40
        )
        guided_epsilon = st.number_input(
            "Guided-filter epsilon",
            min_value=0.000001,
            max_value=1.0,
            value=0.001,
            format="%.6f",
        )
    return {
        "dcp": {
            "patch_size": patch_size,
            "omega": omega,
            "t_min": t_min,
            "atmospheric_percentile": atmospheric_percentile,
            "guided_radius": guided_radius,
            "guided_epsilon": guided_epsilon,
        }
    }


def _show_metrics(st: Any, result: DehazingResult, reference: np.ndarray | None) -> None:
    """Display timing and reference metrics for one result."""
    st.caption(f"Processing time: **{result.processing_time:.4f} seconds**")
    if reference is None:
        st.info("Upload a clear reference image to calculate PSNR and SSIM.")
        return
    try:
        psnr = calculate_psnr(reference, result.image)
        ssim = calculate_ssim(reference, result.image)
    except (TypeError, ValueError) as exc:
        st.warning(f"Reference metrics unavailable: {exc}")
        return
    columns = st.columns(2)
    columns[0].metric("PSNR", "∞ dB" if np.isinf(psnr) else f"{psnr:.2f} dB")
    columns[1].metric("SSIM", f"{ssim:.4f}")


def _result_panel(
    st: Any, heading: str, result: DehazingResult, reference: np.ndarray | None
) -> None:
    """Render one result, its metrics, and a download button."""
    st.subheader(heading)
    st.image(result.image, caption=f"{result.method.upper()} output", use_container_width=True)
    _show_metrics(st, result, reference)
    st.download_button(
        f"Download {result.method.upper()} PNG",
        data=_image_bytes(result.image),
        file_name=f"{result.method}_dehazed.png",
        mime="image/png",
        key=f"download-{result.method}",
    )


def main() -> None:
    """Run the Streamlit dehazing application."""
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is required. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    st.set_page_config(
        page_title="Image Dehazing: Classical vs Deep Learning",
        page_icon="🌫️",
        layout="wide",
    )
    st.title("Image Dehazing: Classical vs Deep Learning")
    st.write(
        "Compare the classical Dark Channel Prior (DCP) method with an optional "
        "pretrained AOD-Net checkpoint. No model training is performed."
    )

    with st.sidebar:
        st.header("Controls")
        method = st.radio(
            "Dehazing method",
            ("Classical DCP", "Pretrained AOD-Net", "Compare Both"),
        )
        device = st.selectbox("Execution device", ("auto", "cpu", "cuda"), index=0)
        checkpoint_text = st.text_input(
            "AOD-Net checkpoint path",
            value="checkpoints/aodnet.pth",
            help="A compatible verified state-dict checkpoint is required for AOD-Net.",
        )
        try:
            resolved_device = select_device(device)
        except RuntimeError as exc:
            resolved_device = None
            st.warning(str(exc))
        if resolved_device is not None:
            st.caption(f"Selected execution device: `{resolved_device}`.")
    dcp_config = _dcp_config(st)

    uploaded = st.file_uploader(
        "Upload a hazy image",
        type=["jpg", "jpeg", "png"],
        help="Images are processed as RGB and retain their original dimensions.",
    )
    reference_file = st.file_uploader(
        "Optional clear-reference image",
        type=["jpg", "jpeg", "png"],
        help="Use an aligned clear image with the same dimensions for PSNR and SSIM.",
    )
    if uploaded is None:
        st.info("Upload a JPG, JPEG, or PNG image to begin.")
        with st.expander("Usage instructions"):
            st.markdown(
                "1. Upload a hazy image.  \n"
                "2. Choose DCP, AOD-Net, or comparison mode.  \n"
                "3. Adjust DCP parameters in the sidebar.  \n"
                "4. Optionally upload an aligned clear reference for PSNR/SSIM.  \n"
                "5. Download each generated PNG from its result panel."
            )
        return

    try:
        image = _uploaded_to_rgb(uploaded)
        reference = _uploaded_to_rgb(reference_file) if reference_file else None
    except ValueError as exc:
        st.error(str(exc))
        return

    st.subheader("Original hazy image")
    st.image(image, caption=f"RGB input: {image.shape[1]} × {image.shape[0]}", use_container_width=True)
    if reference is not None:
        st.caption("Reference metrics require matching dimensions; mismatches are reported below.")

    checkpoint = Path(checkpoint_text).expanduser() if checkpoint_text.strip() else None
    if st.button("Run dehazing", type="primary"):
        if method == "Classical DCP":
            try:
                result = dehaze_with_dcp(image, dcp_config)
            except (TypeError, ValueError, OSError) as exc:
                st.error(f"DCP inference failed: {exc}")
                return
            _result_panel(st, "Classical DCP result", result, reference)
        elif method == "Pretrained AOD-Net":
            try:
                result = dehaze_with_aodnet(image, checkpoint, device)
            except FileNotFoundError as exc:
                st.warning(f"AOD-Net is unavailable: {exc}")
                st.info("Install a compatible checkpoint or select Classical DCP.")
            except (RuntimeError, TypeError, ValueError) as exc:
                st.error(f"AOD-Net inference failed: {exc}")
            else:
                _result_panel(st, "Pretrained AOD-Net result", result, reference)
        else:
            try:
                results = compare_dehazing_methods(image, checkpoint, dcp_config, device)
            except (TypeError, ValueError, OSError) as exc:
                st.error(f"Comparison could not start: {exc}")
                return
            dcp_result = results["dcp"]
            if dcp_result is not None:
                _result_panel(st, "Classical DCP result", dcp_result, reference)
            aod_result = results["aodnet"]
            if aod_result is not None:
                _result_panel(st, "Pretrained AOD-Net result", aod_result, reference)
            else:
                error = results.get("errors", {}).get("aodnet", "checkpoint is unavailable")
                if "not found" in error.lower():
                    st.warning(f"AOD-Net is unavailable: {error}")
                else:
                    st.error(f"AOD-Net inference failed: {error}")


if __name__ == "__main__":
    main()
