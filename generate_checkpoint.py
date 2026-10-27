"""Generate a valid AOD-Net checkpoint with the correct architecture.

Since no publicly-hosted direct-download URL exists for a verified AOD-Net
.pth file, this script creates a checkpoint with properly initialized
(Xavier-uniform) weights. The architecture matches the project's AODNet
class exactly, so inference will run without errors.

For best dehazing quality, replace this checkpoint with one trained on
a real dataset (e.g. NYU Depth, RESIDE ITS/OTS).
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
    checkpoint_path = checkpoint_dir / "aodnet.pth"

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
    print("[OK] Checkpoint verified: loads and produces correct output shape")


if __name__ == "__main__":
    main()
