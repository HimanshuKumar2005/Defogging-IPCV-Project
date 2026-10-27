"""Tests for the checkpoint-compatible AOD-Net inference components."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dehazing.models.aodnet import AODNet
from dehazing.models.pretrained_loader import (
    CheckpointError,
    load_pretrained_model,
    predict,
)


def test_aodnet_preserves_tensor_shape() -> None:
    model = AODNet().eval()
    image = torch.rand(2, 3, 16, 20)
    with torch.inference_mode():
        result = model(image)
    assert result.shape == image.shape
    assert torch.isfinite(result).all()


def test_checkpoint_state_dict_loads_in_eval_mode(tmp_path) -> None:
    source = AODNet()
    checkpoint = tmp_path / "aodnet.pth"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    loaded = load_pretrained_model(checkpoint, "cpu")
    assert not loaded.training
    output = predict(loaded, torch.rand(1, 3, 8, 8))
    assert output.shape == (1, 3, 8, 8)


def test_missing_checkpoint_is_explicit(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_pretrained_model(tmp_path / "missing.pth")


@pytest.mark.parametrize(
    "payload",
    [
        {"state_dict": {"conv1.weight": torch.ones(1)}},
        {"unexpected": torch.ones(1)},
        ["not", "a", "mapping"],
    ],
)
def test_invalid_checkpoint_is_rejected(tmp_path, payload) -> None:
    checkpoint = tmp_path / "invalid.pth"
    torch.save(payload, checkpoint)
    with pytest.raises(CheckpointError):
        load_pretrained_model(checkpoint)


def test_random_weights_are_not_presented_as_loaded(tmp_path) -> None:
    checkpoint = tmp_path / "corrupt.pth"
    checkpoint.write_bytes(b"not a torch checkpoint")
    with pytest.raises(CheckpointError):
        load_pretrained_model(checkpoint)


def test_input_validation() -> None:
    model = AODNet().eval()
    with pytest.raises(ValueError):
        model(torch.rand(1, 1, 8, 8))
    with pytest.raises(TypeError):
        predict(np.zeros((1, 3, 8, 8), dtype=np.float32), torch.rand(1, 3, 8, 8))
