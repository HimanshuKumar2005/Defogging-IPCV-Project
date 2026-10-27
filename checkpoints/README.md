# Checkpoints

# AOD-Net checkpoints

## Verification status

No verified PyTorch checkpoint is bundled. The inspected sources are:

- [walsvid/AOD-Net-PyTorch](https://github.com/walsvid/AOD-Net-PyTorch), whose
  README describes a pretrained PyTorch pickle but leaves its download link
  empty.
- [Boyiliee/AOD-Net](https://github.com/Boyiliee/AOD-Net), which provides the
  original Caffe `AOD_Net.caffemodel` and Caffe inference graph, not a
  checkpoint proven compatible with the PyTorch state-dict loader.

The repositories do not declare an explicit open-source license. Treat their
code and weights as research material, retain the required paper citation, and
obtain permission before redistribution or commercial use. The project does
not convert the Caffe weights or guess that unrelated PyTorch checkpoints are
compatible.

The referenced PyTorch test code uses `PIL.Image` followed by
`torchvision.transforms.ToTensor()`: RGB images are converted to `float32`
tensors in `[0, 1]` with shape `(N, 3, H, W)`. Its example resizes inputs to
`480 x 640`, but the fully convolutional architecture preserves any positive
height and width, so this project does not resize by default. Keep the
checkpoint's original preprocessing convention when comparing published
results.

## Manual checkpoint procedure

When a compatible PyTorch state-dict checkpoint is obtained from the authors or
another trusted source, save it as:

```text
checkpoints/aodnet.pth
```

It must contain the state dictionary for the five-convolution `AODNet`
architecture implemented in
[`src/dehazing/models/aodnet.py`](../src/dehazing/models/aodnet.py), either as
the root mapping or under a `state_dict` key. Verify its provenance, checksum,
license, and expected preprocessing before use. The loader uses
`weights_only=True`, strict key/shape validation, CPU staging, and evaluation
mode. Missing or incompatible files fail explicitly.

Constructing `AODNet()` directly creates an uninitialized architecture for
inspection or checkpoint creation only. It must not be used to produce valid
dehazing results; call `load_pretrained_model()` first.

The application must continue to use DCP when this optional file is absent.
