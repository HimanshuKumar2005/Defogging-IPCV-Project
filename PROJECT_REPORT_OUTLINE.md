# Project Report Outline

## Abstract

State the problem of single-image haze removal and summarize the comparison
between a hand-designed Dark Channel Prior pipeline and inference using a
pretrained AOD-Net. Clearly state that this project does not train a neural
network.

## 1. Introduction

- Define image dehazing.
- Explain why haze affects computer vision and human interpretation.
- Introduce the goal of comparing classical and pretrained deep-learning
  approaches on the same inputs.

## 2. Objectives

- Implement DCP with atmospheric-light and transmission estimation.
- Refine transmission with guided filtering.
- Integrate a compatible pretrained AOD-Net checkpoint when available.
- Provide CPU-safe inference, visual comparison, timing, and evaluation.
- Avoid training and avoid claiming results without valid data.

## 3. Literature review

- Dark Channel Prior by He, Sun, and Tang.
- Guided Image Filtering by He, Sun, and Tang.
- AOD-Net by Li et al.
- Briefly compare explicit priors with learned image-to-image mappings.

## 4. Methodology

### 4.1 Atmospheric scattering model

Describe `I(x) = J(x)t(x) + A(1 - t(x))`, define each term, and explain the
minimum transmission safeguard.

### 4.2 Classical DCP

Describe local dark-channel calculation, atmospheric-light selection, coarse
transmission, guided refinement, and scene-radiance recovery.

### 4.3 Pretrained AOD-Net

Describe the five-convolution architecture, feature concatenations, RGB
`[0, 1]` preprocessing, checkpoint-only inference, and evaluation mode.

### 4.4 Unified inference

Explain device selection, RGB/BGR conversion, model caching, timing boundaries,
missing-checkpoint behavior, and output clipping.

## 5. System architecture

Include a diagram or describe:

```text
Input image
    ├── RGB validation and normalization
    ├── DCP ──> dark channel ──> transmission ──> radiance
    └── AOD-Net checkpoint ──> tensor inference ──> RGB output
                 |
                 └── metrics, timing, visualization, CSV reports
```

Reference the modules:

- `app.py`
- `src/dehazing/classical/dcp.py`
- `src/dehazing/models/aodnet.py`
- `src/dehazing/models/pretrained_loader.py`
- `src/dehazing/inference.py`
- `src/dehazing/metrics.py`
- `scripts/batch_process.py`
- `scripts/evaluate.py`

## 6. Implementation

- Python, OpenCV, NumPy, PyTorch, Pillow, scikit-image, Streamlit,
  matplotlib, and pandas.
- Configuration from `configs/default.yaml`.
- CPU fallback and optional CUDA.
- Safe state-dict loading with strict validation.
- Per-image status and error reporting.

## 7. Experimental setup

Document:

- Python version and dependency versions used.
- CPU/GPU hardware.
- Image dimensions and color convention.
- DCP parameter values.
- AOD-Net checkpoint source, checksum, and license.
- Paired dataset name, subset, license, and manifest.
- Timing definition and warm-up/model-loading policy.

If a real checkpoint or paired dataset was not available, state that explicitly
instead of filling this section with invented values.

## 8. Results and comparison

Insert only measured results from `comparison.csv` and `summary.csv`:

- Per-image PSNR and SSIM.
- Mean and standard deviation by method.
- Inference time by method.
- Before/after and difference images.
- Dark-channel and transmission visualizations for DCP.

Do not claim one method is better without comparing the same images, valid
references, and compatible preprocessing.

## 9. Limitations

- Bright skies and bright objects can violate DCP assumptions.
- DCP may create halos or color shifts.
- AOD-Net depends on checkpoint provenance and training domain.
- Synthetic-training domain and real-world haze may differ.
- PSNR/SSIM do not fully represent perceptual quality.
- CPU-only execution can be slower.
- No verified checkpoint or paired dataset may be available in a clean
  installation.

## 10. Conclusion and future scope

Summarize what was implemented and what was measured. State that the current
system performs inference with a pretrained AOD-Net only; it does not train a
neural network. Future work may include:

- Fine-tuning AOD-Net on a licensed domain-specific dataset.
- Training AOD-Net from scratch.
- More robust real-world haze datasets.
- Additional perceptual and no-reference metrics.
- User studies and latency profiling on GPU hardware.
