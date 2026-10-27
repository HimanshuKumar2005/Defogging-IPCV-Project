"""Tests for the classical Dark Channel Prior implementation."""

from __future__ import annotations

import numpy as np
import pytest

from dehazing.classical.dcp import (
    dehaze_dcp,
    estimate_atmospheric_light,
    estimate_transmission,
    get_dark_channel,
    recover_scene_radiance,
    refine_transmission,
)


@pytest.fixture
def sample_image() -> np.ndarray:
    image = np.full((24, 32, 3), 0.35, dtype=np.float32)
    image[6:18, 8:24] = [0.8, 0.7, 0.6]
    return image


def test_dark_channel_uses_local_minimum(sample_image: np.ndarray) -> None:
    dark = get_dark_channel(sample_image, 3)
    assert dark.shape == sample_image.shape[:2]
    assert np.isclose(dark[12, 16], 0.6)
    assert dark.min() >= 0


def test_atmospheric_light_is_bright_candidate(sample_image: np.ndarray) -> None:
    dark = get_dark_channel(sample_image, 3)
    atmospheric = estimate_atmospheric_light(sample_image, dark)
    assert atmospheric.shape == (3,)
    assert np.allclose(atmospheric, [0.8, 0.7, 0.6])


def test_transmission_and_refinement_are_bounded(sample_image: np.ndarray) -> None:
    atmospheric = np.array([0.8, 0.7, 0.6], dtype=np.float32)
    transmission = estimate_transmission(sample_image, atmospheric, 0.95, 5)
    refined = refine_transmission(sample_image, transmission, 2, 1e-3)
    assert transmission.shape == refined.shape == sample_image.shape[:2]
    assert np.isfinite(refined).all()
    assert np.all((refined >= 0) & (refined <= 1))


def test_recover_scene_radiance_is_numerically_stable(
    sample_image: np.ndarray,
) -> None:
    transmission = np.zeros(sample_image.shape[:2], dtype=np.float32)
    result = recover_scene_radiance(
        sample_image, transmission, np.array([0.8, 0.7, 0.6]), 0.1
    )
    assert result.shape == sample_image.shape
    assert np.isfinite(result).all()
    assert np.all((result >= 0) & (result <= 1))


def test_dehaze_preserves_dimensions_and_supports_uint8(
    sample_image: np.ndarray,
) -> None:
    result = dehaze_dcp(sample_image, {"dcp": {"patch_size": 5, "t_min": 0.2}})
    uint8_result = dehaze_dcp((sample_image * 255).astype(np.uint8))
    assert result.shape == sample_image.shape
    assert uint8_result.shape == sample_image.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


@pytest.mark.parametrize(
    "bad_image",
    [
        np.zeros((10, 10), dtype=np.float32),
        np.zeros((10, 10, 4), dtype=np.float32),
        np.full((10, 10, 3), np.nan, dtype=np.float32),
        np.full((10, 10, 3), 2.0, dtype=np.float32),
    ],
)
def test_invalid_images_raise_value_error(bad_image: np.ndarray) -> None:
    with pytest.raises((ValueError, TypeError)):
        get_dark_channel(bad_image, 3)


def test_invalid_parameters_raise(sample_image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        get_dark_channel(sample_image, 0)
    with pytest.raises(ValueError):
        estimate_transmission(sample_image, np.ones(3), 1.1, 3)
    with pytest.raises(ValueError):
        refine_transmission(sample_image, np.ones((24, 32)), 0, 1e-3)
