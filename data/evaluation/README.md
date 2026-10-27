# Evaluation data

Place evaluation image pairs here when running quantitative comparisons.
Keep hazy inputs and corresponding clear references organized consistently, for
example:

```text
data/evaluation/
├── hazy/
└── clear/
```

No training dataset is required by this project version.

## Evaluation manifest

Evaluation uses a CSV manifest so the same hazy inputs are passed to every
selected method. Required columns are `image_path` and `reference_path`;
`image_id` is optional:

```csv
image_id,image_path,reference_path
scene_01,hazy/scene_01.png,clear/scene_01.png
scene_02,hazy/scene_02.png,clear/scene_02.png
```

Paths are resolved relative to the manifest file. Leave `reference_path` empty
when no aligned clear reference is available; PSNR and SSIM will then be
recorded as unavailable rather than calculated against the hazy input.

Run evaluation without downloading a dataset:

```powershell
$env:PYTHONPATH = (Get-Location).Path + "\src"
python scripts\evaluate.py data\evaluation\manifest.csv `
  --output-dir outputs\evaluation --method both --device cpu --visuals --plot
```

For a small paired subset, obtain a suitably licensed RESIDE subset through its
official distribution, select a few hazy/clear pairs, copy only those images
under `data/evaluation/hazy` and `data/evaluation/clear`, and create the
manifest above. Do not commit the dataset or download the full dataset
automatically. Confirm the dataset license and preserve its attribution.
