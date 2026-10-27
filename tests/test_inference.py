"""Tests for the shared DCP/AOD-Net inference interface."""

from __future__ import annotations

import numpy as np
import torch

from dehazing.inference import (
    _MODEL_CACHE,
    compare_dehazing_methods,
    dehaze_with_aodnet,
    dehaze_with_dcp,
    run_inference,
)
from dehazing.models.aodnet import AODNet


def _image() -> np.ndarray:
    return np.full((12, 16, 3), 0.4, dtype=np.float32)


def test_dcp_result_contract_and_dimensions() -> None:
    result = dehaze_with_dcp(_image(), {"dcp": {"patch_size": 3, "guided_radius": 2}})
    assert result.method == "dcp"
    assert result.image.shape == _image().shape
    assert result.image.dtype == np.float32
    assert result.processing_time >= 0
    assert np.isfinite(result.image).all()


def test_aodnet_result_contract_and_model_cache(tmp_path) -> None:
    checkpoint = tmp_path / "aodnet.pth"
    torch.save({"state_dict": AODNet().state_dict()}, checkpoint)
    _MODEL_CACHE.clear()
    first = dehaze_with_aodnet(_image(), checkpoint, "cpu")
    second = dehaze_with_aodnet(_image(), checkpoint, "cpu")
    assert first.method == second.method == "aodnet"
    assert first.image.shape == _image().shape
    assert first.metadata["device"] == "cpu"
    assert first.metadata["resized"] is False
    assert len(_MODEL_CACHE) == 1


def test_missing_aod_checkpoint_does_not_block_dcp() -> None:
    results = compare_dehazing_methods(
        _image(), r"C:\does-not-exist\aodnet.pth", {"dcp": {"patch_size": 3}}
    )
    assert results["dcp"] is not None
    assert results["dcp"].image.shape == _image().shape
    assert results["aodnet"] is None
    assert "aodnet" in results["errors"]


def test_run_inference_rejects_unknown_method() -> None:
    try:
        run_inference(_image(), "unknown")
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("unknown methods must fail explicitly")
