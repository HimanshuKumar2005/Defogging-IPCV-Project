"""Tests for Streamlit presentation helpers using actual image data."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

import app


def test_image_bytes_round_trip_preserves_rgb_pixels() -> None:
    image = np.zeros((5, 6, 3), dtype=np.uint8)
    image[..., 0] = 255
    decoded = np.asarray(Image.open(BytesIO(app._image_bytes(image))).convert("RGB"))
    assert decoded.shape == image.shape
    assert np.array_equal(decoded, image)


def test_image_bytes_encodes_float_outputs() -> None:
    image = np.full((4, 4, 3), 0.5, dtype=np.float32)
    decoded = np.asarray(Image.open(BytesIO(app._image_bytes(image))).convert("RGB"))
    assert decoded[0, 0, 0] == 128


def test_uploaded_file_conversion_is_rgb() -> None:
    buffer = BytesIO()
    Image.fromarray(np.array([[[10, 20, 30]]], dtype=np.uint8), mode="RGB").save(
        buffer, format="PNG"
    )
    buffer.seek(0)
    converted = app._uploaded_to_rgb(buffer)
    assert converted.shape == (1, 1, 3)
    assert converted[0, 0].tolist() == [10, 20, 30]
