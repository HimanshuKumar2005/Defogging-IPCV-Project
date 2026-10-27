# Image Dehazing: Classical Image Processing vs Deep Learning

This project compares two single-image dehazing approaches:

1. A classical Dark Channel Prior (DCP) pipeline implemented in Python.
2. A pretrained AOD-Net inference path, when a compatible checkpoint is available.

This version is an initial project scaffold. It intentionally does **not** train a
neural network, require a training dataset, or include a training pipeline.
Classical processing and the application are designed to remain usable when the
AOD-Net checkpoint is absent.

## Project layout

```text
.
├── app.py
├── configs/
│   └── default.yaml
├── checkpoints/
│   └── README.md
├── data/evaluation/
│   └── README.md
├── outputs/
├── scripts/
│   ├── batch_process.py
│   └── evaluate.py
├── src/dehazing/
│   ├── classical/dcp.py
│   ├── inference.py
│   ├── metrics.py
│   ├── models/
│   │   ├── aodnet.py
│   │   └── pretrained_loader.py
│   └── utils/image_io.py
├── tests/
├── requirements.txt
└── README.md
```

## Environment setup

From the project root on Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CPU execution is supported. PyTorch will use CUDA automatically when available
once the application and inference modules are implemented.

## Current status

The classical DCP pipeline is now available. It accepts RGB `uint8` images or
floating-point images in `[0, 1]` and returns RGB `float32` output in `[0, 1]`.
For example, after installing dependencies:

```powershell
$env:PYTHONPATH = (Get-Location).Path + "\src"
python -c "import cv2; from dehazing.classical.dcp import dehaze_dcp; image = cv2.cvtColor(cv2.imread('input.jpg'), cv2.COLOR_BGR2RGB); result = dehaze_dcp(image); cv2.imwrite('outputs/dehazed.png', cv2.cvtColor((result * 255).astype('uint8'), cv2.COLOR_RGB2BGR))"
```

The implementation uses OpenCV's `ximgproc.guidedFilter` when available and a
vectorized grayscale guided-filter fallback otherwise. The checkpoint-compatible
AOD-Net architecture and safe optional loader are available, but no verified
PyTorch checkpoint is bundled because the referenced repositories do not expose
one at a stable, verifiable source. See [checkpoints/README.md](checkpoints/README.md).
Both methods are available through the central inference interface. For a
single image:

```powershell
$env:PYTHONPATH = (Get-Location).Path + "\src"
python -m dehazing.inference input.jpg --method dcp --output-path outputs\dcp.png
python -m dehazing.inference input.jpg --method aodnet --checkpoint checkpoints\aodnet.pth --output-path outputs\aodnet.png
```

The CLI reads BGR files with OpenCV, converts them to RGB for processing, and
converts results back to BGR when saving. Batch processing and manifest-driven
evaluation are available through [scripts/batch_process.py](scripts/batch_process.py)
and [scripts/evaluate.py](scripts/evaluate.py). See
[data/evaluation/README.md](data/evaluation/README.md) for a small paired-subset
workflow; no dataset is downloaded automatically.

The Streamlit application provides before/after and three-panel displays,
optional zoomed center crops, downloadable PNG outputs, reference difference
views, DCP dark-channel and transmission diagnostics, and a multi-file batch
summary. Every displayed output and metric comes from the selected algorithm's
actual execution; missing AOD-Net weights remain an explicit unavailable state.

## Quality metrics

[metrics.py](src/dehazing/metrics.py) provides PSNR and RGB SSIM for comparing
an output against a genuine, aligned clear reference image. Images are
normalized to `[0, 1]`, so both metrics use `data_range=1.0`. Identical images
have infinite PSNR and SSIM of 1.0. If no reference is available, the metrics
are recorded as unavailable (`None`/blank in CSV), never calculated against the
hazy input or a fabricated image. Per-image records can be aggregated with
mean and population standard deviation and exported to CSV. Inference timing
starts after model loading; CUDA timing synchronizes before and after the
measured call.
