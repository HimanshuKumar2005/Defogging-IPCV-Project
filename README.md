# Image Dehazing: Classical Image Processing vs Deep Learning

## 1. Project objective and problem statement

Haze reduces visibility by scattering and attenuating light. This project
implements an inference-only comparison of:

1. A classical Dark Channel Prior (DCP) dehazing pipeline implemented in this
   repository.
2. A pretrained PyTorch AOD-Net model, when a compatible checkpoint is
   available.

The goal is to demonstrate how the two approaches process the same RGB image,
measure their execution time, and evaluate them against a genuine clear
reference when paired data is available. The project does **not** train a
neural network, require a training dataset, or claim that either method is
universally superior.

## 2. Motivation

DCP is interpretable and does not require learned weights, making it useful on
CPU-only systems and when no model checkpoint is available. AOD-Net is a
lightweight learned model that can produce different restoration behavior
after training. Comparing them is educational because it shows the trade-offs
between a hand-designed image prior and a pretrained neural network:

- DCP exposes intermediate quantities such as the dark channel and
  transmission map.
- AOD-Net performs a learned end-to-end mapping after checkpoint loading.
- PSNR, SSIM, visual appearance, and runtime can disagree, so all should be
  reported honestly.

## 3. Technical background

### Atmospheric scattering model

A common single-image haze model is:

```text
I(x) = J(x) * t(x) + A * (1 - t(x))
```

where:

- `I(x)` is the observed hazy image.
- `J(x)` is the unknown clear scene radiance.
- `t(x)` is the transmission describing how much scene light reaches the
  camera.
- `A` is the global atmospheric light.

Dehazing estimates `A` and `t(x)`, then recovers `J(x)` using a minimum
transmission value to avoid division by zero.

### Dark Channel Prior

The DCP assumes that in most non-sky patches of clear outdoor images at least
one color channel contains a very low value. The implementation:

1. Computes a per-pixel RGB minimum.
2. Applies a local minimum erosion over a configurable patch.
3. Estimates atmospheric light from bright candidates in the dark channel.
4. Estimates coarse transmission from the normalized image.
5. Refines transmission using OpenCV `ximgproc.guidedFilter` when available.
6. Uses a vectorized grayscale guided-filter fallback when `ximgproc` is not
   installed.
7. Recovers and clips the scene radiance to `[0, 1]`.

The DCP implementation is in
[src/dehazing/classical/dcp.py](<C:/Users/Himanshu Kumar/OneDrive/Desktop/2026/codex/Defogging IPCV Project/src/dehazing/classical/dcp.py>).

### AOD-Net

AOD-Net (All-in-One Dehazing Network) is a lightweight fully convolutional
network. The compatible implementation uses five convolutions with the
following channel/kernel structure:

```text
3 -> 3, 1x1
3 -> 3, 3x3
6 -> 3, 5x5
6 -> 3, 7x7
12 -> 3, 3x3
```

Intermediate features are concatenated and the network predicts an
all-in-one transform used to produce the restored image. The implementation
accepts RGB tensors shaped `(N, 3, H, W)` with values in `[0, 1]` and preserves
height and width.

The architecture is implemented in
[src/dehazing/models/aodnet.py](<C:/Users/Himanshu Kumar/OneDrive/Desktop/2026/codex/Defogging IPCV Project/src/dehazing/models/aodnet.py>).
It is inference-only in this project. Constructing `AODNet()` alone creates
uninitialized weights and is never treated as a valid dehazing model.

## 4. Checkpoint source, licensing, and limitations

The inspected reference repositories are:

- [walsvid/AOD-Net-PyTorch](https://github.com/walsvid/AOD-Net-PyTorch)
- [Boyiliee/AOD-Net](https://github.com/Boyiliee/AOD-Net)

The first repository documents a PyTorch checkpoint but currently leaves its
download link empty. The second provides the original Caffe
`AOD_Net.caffemodel` and Caffe inference graph. A Caffe file is **not**
assumed to be compatible with this PyTorch state-dict loader, and no
unverified conversion is performed.

Neither referenced repository declares an explicit open-source license in the
inspected material. Treat their code and weights as research material, retain
the paper citation, verify permission and licensing before redistribution or
commercial use, and record provenance and a checksum for any checkpoint used.

### Exact checkpoint procedure

No verified checkpoint is bundled. If the authors or another trusted source
provides a compatible PyTorch state-dict checkpoint:

1. Verify its source, license, checksum, expected preprocessing, and model
   revision.
2. Save it with this exact filename:

   ```text
   checkpoints\aodnet.pth
   ```

3. The file must contain either a root state-dict mapping or a mapping under
   `state_dict`, with keys and shapes matching
   [aodnet.py](<C:/Users/Himanshu Kumar/OneDrive/Desktop/2026/codex/Defogging IPCV Project/src/dehazing/models/aodnet.py>).
4. Do not rename a Caffe `.caffemodel` file to `.pth`.
5. Do not use a randomly initialized `AODNet()` for results.

The safe loader uses `weights_only=True`, stages weights on CPU, validates the
state dictionary strictly, moves the model to the selected device, and sets
evaluation mode. Missing, corrupted, or incompatible checkpoints fail
explicitly. DCP remains available without the checkpoint.

## 5. Installation and environment setup

From the project root in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path + "\src"
```

Python 3.12 was used for the verified local run. CPU execution is supported.
CUDA is selected automatically when a CUDA-enabled PyTorch installation and
compatible GPU are available; CUDA is never required.

## 6. Launch the Streamlit demonstration

```powershell
$env:PYTHONPATH = (Get-Location).Path + "\src"
python -m streamlit run app.py
```

The application supports:

- JPG, JPEG, and PNG uploads.
- DCP, pretrained AOD-Net, and comparison modes.
- DCP parameter controls.
- CPU/CUDA/automatic device selection.
- Before/after and three-panel displays.
- Optional zoomed crops.
- DCP dark-channel and transmission diagnostics.
- Optional clear-reference PSNR/SSIM.
- Reference difference views.
- Downloadable PNG outputs.
- Multi-file batch upload summaries.

If `checkpoints\aodnet.pth` is absent, choose DCP or use comparison mode; DCP
will continue and the UI will label AOD-Net as unavailable.

## 7. Process one image

The CLI reads files with OpenCV, converts BGR to RGB for processing, and
converts RGB output back to BGR for saving:

```powershell
$env:PYTHONPATH = (Get-Location).Path + "\src"
python -m dehazing.inference input.jpg `
  --method dcp `
  --output-path outputs\dcp\input_dcp.png `
  --device cpu
```

With a verified checkpoint:

```powershell
python -m dehazing.inference input.jpg `
  --method aodnet `
  --checkpoint checkpoints\aodnet.pth `
  --output-path outputs\aodnet\input_aodnet.png `
  --device cpu
```

The output path must differ from the input path.

## 8. Process a directory

```powershell
$env:PYTHONPATH = (Get-Location).Path + "\src"
python scripts\batch_process.py data\input `
  --output-dir outputs\batch `
  --method both `
  --device cpu
```

With AOD-Net:

```powershell
python scripts\batch_process.py data\input `
  --output-dir outputs\batch `
  --method both `
  --checkpoint checkpoints\aodnet.pth `
  --config configs\default.yaml `
  --device cpu
```

Outputs are separated into `outputs\batch\dcp` and
`outputs\batch\aodnet`. Status and timing are written to
`outputs\batch\batch_status.csv`. Original inputs are never overwritten.

## 9. Evaluate a paired test subset

Prepare a small, suitably licensed paired subset manually. Do not download a
large dataset automatically and do not commit the data. A recommended layout
is:

```text
data\evaluation\
├── hazy\
├── clear\
└── manifest.csv
```

Manifest format:

```csv
image_id,image_path,reference_path
scene_01,hazy/scene_01.png,clear/scene_01.png
scene_02,hazy/scene_02.png,clear/scene_02.png
```

Paths are resolved relative to the manifest. For RESIDE or another dataset,
obtain the data from its official distribution, confirm its license, select a
small paired subset, and preserve the original attribution. No training data
is needed for this project.

Run:

```powershell
$env:PYTHONPATH = (Get-Location).Path + "\src"
python scripts\evaluate.py data\evaluation\manifest.csv `
  --output-dir outputs\evaluation `
  --method both `
  --device cpu `
  --visuals `
  --plot
```

The command produces:

- `comparison.csv`: one row per image and method.
- `summary.csv`: method-specific means, standard deviations, and counts.
- Optional method-specific visual outputs.
- Optional `metrics.png`, generated only from actual finite PSNR/SSIM values.

If a reference is missing or invalid, PSNR and SSIM are blank and the status
explains why. If AOD-Net is unavailable, its rows explicitly say
`model_unavailable`; DCP rows remain valid.

## 10. Metrics and timing

- **PSNR** measures pixel-wise error in decibels. Higher is generally better;
  identical images have infinite PSNR.
- **SSIM** compares local luminance, contrast, and structure. Values closer to
  1 indicate greater structural similarity.
- **Inference time** is measured for the actual method execution. AOD-Net
  checkpoint loading and model initialization are excluded from steady-state
  timing. CUDA timing synchronizes before and after the measured call.

PSNR and SSIM are valid only when the output and a genuine aligned clear
reference have matching dimensions. They do not prove perceptual superiority,
and visual quality can disagree with both metrics.

## 11. Known limitations

- DCP can produce halos near strong depth or brightness transitions.
- DCP assumptions are weaker for bright skies, white objects, large bright
  regions, and non-outdoor scenes.
- DCP parameters can strongly affect color and contrast.
- AOD-Net results depend on the exact checkpoint, preprocessing, and training
  domain.
- A model trained on synthetic haze may not generalize to real-world haze.
- No verified AOD-Net checkpoint is bundled, so deep-learning inference is
  unavailable until one is obtained and verified.
- No benchmark values are included because no paired dataset was run here.
- CPU inference is functional but may be slower than GPU inference.
- PSNR and SSIM do not fully capture perceptual quality.

## 12. Troubleshooting

### `ModuleNotFoundError: No module named 'dehazing'`

From the project root, set:

```powershell
$env:PYTHONPATH = (Get-Location).Path + "\src"
```

### Streamlit is not recognized or cannot import

Activate the virtual environment and reinstall dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### AOD-Net checkpoint not found

Place a verified compatible state-dict at
`checkpoints\aodnet.pth`. Do not use the Caffe `.caffemodel` as a PyTorch
checkpoint. DCP does not depend on this file.

### Checkpoint is rejected as incompatible or corrupted

Confirm that the file is a PyTorch state dictionary for the five-convolution
architecture, that it was downloaded completely, and that its provenance and
checksum are trusted. The loader intentionally does not fall back to random
weights.

### CUDA is unavailable

Use `--device cpu`, select CPU in Streamlit, or use automatic device selection.
The project is designed to run on CPU.

### PSNR/SSIM are unavailable

Upload or manifest a genuine clear reference with the same width, height, and
RGB layout as the hazy image and output.

## 13. References

1. Boyi Li, Xiulian Peng, Zhangyang Wang, Jizheng Xu, and Dan Feng,
   “AOD-Net: All-in-One Dehazing Network,” ICCV 2017.
   [Paper](https://openaccess.thecvf.com/content_iccv_2017/html/Li_AOD-Net_All-in-One_Dehazing_ICCV_2017_paper.html)
2. Kaiming He, Jian Sun, and Xiaoou Tang,
   “Single Image Haze Removal Using Dark Channel Prior,” IEEE TPAMI 2011.
   [Paper](https://ieeexplore.ieee.org/document/5567108)
3. Kaiming He, Jian Sun, and Xiaoou Tang,
   “Guided Image Filtering,” IEEE TPAMI 2013.
   [Paper](https://arxiv.org/abs/1505.00996)

The code implementations and repository links are documented in
[checkpoints/README.md](<C:/Users/Himanshu Kumar/OneDrive/Desktop/2026/codex/Defogging IPCV Project/checkpoints/README.md>).

## 14. Final project structure

```text
.
├── app.py
├── configs/default.yaml
├── checkpoints/README.md
├── data/evaluation/README.md
├── outputs/.gitkeep
├── scripts/
│   ├── batch_process.py
│   └── evaluate.py
├── src/dehazing/
│   ├── classical/dcp.py
│   ├── inference.py
│   ├── metrics.py
│   ├── models/aodnet.py
│   ├── models/pretrained_loader.py
│   └── utils/image_io.py
├── tests/
├── requirements.txt
└── README.md
```

## 15. Five-minute college presentation sequence

1. **0:00–0:30 — Problem:** show a hazy input and explain visibility loss.
2. **0:30–1:15 — Model:** introduce the atmospheric scattering equation.
3. **1:15–2:00 — Classical method:** explain DCP, atmospheric light,
   transmission, and guided filtering; show diagnostic maps.
4. **2:00–2:45 — Deep-learning method:** explain AOD-Net’s lightweight
   five-convolution architecture and clarify that this project uses pretrained
   inference only, not training.
5. **2:45–3:45 — Live demonstration:** upload one image, run DCP, show the
   before/after and timing, then show AOD-Net availability status.
6. **3:45–4:30 — Evaluation:** show a manifest, per-image CSV, PSNR/SSIM,
   timing, and a comparison table only if a real checkpoint and paired data
   are available.
7. **4:30–5:00 — Limitations/future work:** discuss bright skies, halos,
   domain shift, and future fine-tuning or training from scratch.

## 16. Verification status

The verified local environment used Python 3.12.10, PyTorch 2.6.0 CPU-only,
OpenCV, NumPy, Pillow, PyYAML, scikit-image, matplotlib, pandas, Streamlit,
torchvision, and pytest. The complete automated suite passed:

```text
36 passed
```

Streamlit startup and its health endpoint also passed. CUDA inference, a real
pretrained AOD-Net checkpoint, and a real paired dataset were not available in
that environment, so no benchmark claim is made for them.
