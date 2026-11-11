"""Generate an architecture-only AOD-Net checkpoint for tests and inspection.

This script does not create a dehazing model. Randomly initialized weights
cannot produce meaningful restoration results and must not be used by the
application.

Use a verified checkpoint trained on a real dataset (e.g. NYU Depth or
RESIDE ITS/OTS) for actual inference.
"""
import sys
from pathlib import Path

# Add src/ so we can import the project's own AODNet class
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from torch.nn import init
from src.dehazing.models.aodnet import AODNet


def main() -> None:
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "aodnet_untrained_for_tests.pth"

    model = AODNet()

    # Xavier-uniform init gives much better starting results than default
    # kaiming_uniform (PyTorch default for Conv2d) for this architecture.
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)

    state_dict = model.state_dict()
    torch.save({"state_dict": state_dict}, checkpoint_path)
    print(f"[OK] AOD-Net checkpoint saved to: {checkpoint_path}")
    print(f"  Keys: {list(state_dict.keys())}")
    print(f"  File size: {checkpoint_path.stat().st_size / 1024:.1f} KB")

    # Verify it loads correctly through the project's own loader
    from src.dehazing.models.pretrained_loader import load_pretrained_model
    loaded = load_pretrained_model(checkpoint_path, "cpu")
    assert not loaded.training, "Model should be in eval mode"

    # Quick sanity: forward pass
    dummy = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        out = loaded(dummy)
    assert out.shape == dummy.shape, f"Shape mismatch: {out.shape} vs {dummy.shape}"
    print("[OK] Architecture checkpoint verified; do not use it for dehazing.")


if __name__ == "__main__":
    main()
