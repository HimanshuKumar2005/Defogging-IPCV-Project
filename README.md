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
vectorized grayscale guided-filter fallback otherwise. AOD-Net inference,
Streamlit UI, batch processing, and reporting remain future work.
