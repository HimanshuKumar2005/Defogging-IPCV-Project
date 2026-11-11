# Preliminary dehazing evaluation — 11 November 2026

## Latest result: 100 test scenes

| Method | Mean PSNR (dB) | Mean SSIM | Structural similarity (SSIM × 100) |
| --- | ---: | ---: | ---: |
| Untreated hazy input | 15.85 | 0.8182 | 81.82% |
| Default DCP | 18.06 | 0.8934 | 89.34% |
| Frozen tuned DCP | **22.05** | **0.9263** | **92.63%** |
| Pretrained AOD-Net candidate | 19.71 | 0.9011 | 90.11% |

All 100 scenes completed for all four methods (400 metric rows), with no
skipped or failed images. The original eight-scene results reproduced within
the numerical tolerance check. Tuned DCP's 95% scene-bootstrap interval is
91.91–93.29% structural similarity. On the 92 newly added scenes alone,
tuned DCP achieved 92.74% structural similarity and 22.06 dB PSNR.

Compared with default DCP on these same 100 scenes, the frozen tuned preset
gains 3.29 percentage points of structural similarity and 3.99 dB PSNR.
The change from 91.36% on eight scenes to 92.63% on 100 scenes reflects the
larger test sample; no additional model improvement or training occurred.
These percentages are scaled SSIM scores, not classification accuracy.

## Expanded evaluation protocol

The expanded run increases the test set from 8 to **100 distinct outdoor
scenes**. The original 8 test scenes are retained; 92 additional scenes are
sampled without replacement with Python random seed 42 from the remaining
sorted scene list. Each scene uses its first available haze variant in
lexicographic filename order. The same 4 validation scenes are excluded.
The DCP settings selected in the initial experiment and the AOD-Net checkpoint
are frozen; there is no tuning on the expanded test set.

The expanded downloader verifies ZIP CRCs and records SHA-256 hashes. The
evaluator checks unique scene IDs, validation exclusion, checkpoint identity,
finite scores, matching image dimensions, and reproduction of the original
eight-scene scores. It reports all 100 scenes and the 92 new scenes separately.
Bootstrap intervals use 10,000 scene resamples (seed 42); they describe sample
variability, not performance on every real-world haze domain. This remains a
subset evaluation from a third-party SOTS mirror, not a full benchmark.

```powershell
& outputs/evaluation_setup/python/python.exe scripts/download_sots_sample.py --test-count 100
& outputs/evaluation_setup/python/python.exe scripts/evaluate_expanded.py --test-count 100
```

Expanded outputs: `outputs/sots_100/comparison.csv` and
`outputs/sots_100/results.json`. Dataset manifest and provenance:
`data/evaluation/sots_sample/manifest_100.csv` and `provenance_100.json`.
The percentage column is **100 × mean SSIM**, which is structural similarity
expressed as a percentage, not classification accuracy.

This run evaluates the existing DCP implementation and a newly downloaded,
third-party pretrained AOD-Net candidate. The repository originally had no
checkpoint or evaluation images. These results are a small-sample diagnostic,
not a full RESIDE benchmark or a percentage accuracy score.

## Measured results

Mean scores on eight test scenes, excluded from the DCP parameter search:

| Method | PSNR (dB) | SSIM |
| --- | ---: | ---: |
| Untreated hazy input | 15.9340 | 0.7991 |
| Existing default DCP | 18.3420 | 0.8924 |
| Validation-tuned DCP | **21.9132** | **0.9136** |
| Newly obtained pretrained AOD-Net candidate | 18.9696 | 0.8786 |

Tuned DCP gains **3.5713 dB PSNR** and **0.0212 SSIM** over default DCP.
The selected settings are saved in `configs/dcp_sots_sample.yaml`:
patch size 31, omega 0.75, minimum transmission 0.08, atmospheric percentile
0.001, guided radius 20, guided epsilon 0.005. This is an experimental preset;
it has not been promoted to the application default on eight scenes alone.

The source-equation equivalence check passed. The existing targeted test suite
passed: **27 tests** covering metrics, DCP, and AOD-Net. Dependency versions are
recorded in `outputs/sots_sample/requirements-lock.txt`.
Times in the diagnostic CSV are wall-clock call durations and include first-call
model initialization where applicable; do not interpret them as a formal speed benchmark.

## Protocol

- Dataset: 12 distinct outdoor scenes, one haze variant each, from a
  [RESIDE-SOTS mirror](https://huggingface.co/datasets/desonglll/reside-SOTS),
  pinned to revision `04dd3b3eb281637d2e3a0232368c43fce846e505`.
  [Original dataset project](https://github.com/Boyiliee/RESIDE-dataset-link).
- Select the first 12 distinct scenes in lexicographic filename order, before
  measuring scores. This convenience sample is not statistically representative.
- Validation scenes: 0001, 0002, 0003, 0004. Test scenes: 0006, 0007, 0009,
  0010, 0011, 0014, 0016, 0017. These are local diagnostic splits within SOTS;
  do not report validation-tuned results as an untouched official SOTS score.
- Evaluate original image dimensions and RGB channels, without resizing or
  cropping. PSNR uses data range 1; SSIM uses the project's scikit-image
  implementation with a 7-pixel window. Average scores per image.
- Compare the untreated hazy input, default DCP, validation-tuned DCP, and
  AOD-Net. Tune 18 DCP combinations using validation mean PSNR only:
  patch size {7,15,31}, omega {0.75,0.90,0.98}, minimum transmission {0.08,0.16}.
  Other settings retain their existing defaults.
- Checkpoint: [MayankSingal/PyTorch-Image-Dehazing](https://github.com/MayankSingal/PyTorch-Image-Dehazing),
  `snapshots/dehazer.pth`. The source repository carries an MIT license; a copy
  is retained in `outputs/evaluation_setup/checkpoint_LICENSE`.
  Load with `weights_only=True`, require exactly the ten expected tensors,
  map `e_conv1..5` to `conv1..5`, and load strictly. Verify bit-exact forward
  equivalence against the source equations on a seeded input.
  This verifies architectural compatibility, not training provenance or
  absence of training/test overlap; the checkpoint's training history is not audited.
- Downloads use partial ZIP requests and CRC checks. Individual SHA-256 hashes
  and source URL are saved in `data/evaluation/sots_sample/provenance.json`.
  These checks establish file integrity, not independent authentication of the mirror.

## Outputs and reproduction

The runner is `scripts/benchmark_sample.py`; download preparation is
`scripts/download_sots_sample.py`. Run from the repository root. The portable
Windows Python and its evaluation dependencies are under
`outputs/evaluation_setup/python/`, because the existing MSYS Python cannot
use standard Windows PyTorch wheels.

```powershell
& outputs/evaluation_setup/python/python.exe scripts/benchmark_sample.py
```

The original downloaded checkpoint is required at
`outputs/evaluation_setup/dehazer.pth`. The runner writes per-image scores to
`outputs/sots_sample/comparison.csv`, aggregate results and all validation
trials to `outputs/sots_sample/results.json`, PNGs to
`outputs/sots_sample/images`, and the compatible checkpoint to
`outputs/sots_sample/aodnet_compatible.pth`. Application defaults are unchanged.

## Priorities for improving quality

1. Establish a larger, scene-disjoint training/validation/test protocol matched
   to the intended images. Preserve a completely untouched test set; keep all
   haze variants of one scene in the same split. Include untreated inputs as a
   baseline and inspect per-scene failures, especially bright skies and dense haze.
2. Tune DCP on validation data, including patch size, omega, minimum transmission,
   guided-filter radius/epsilon, and atmospheric-light candidate fraction.
   Promote settings only if they generalize across a larger separate test set.
3. Benchmark a stronger pretrained model such as DehazeFormer with the correct
   domain-specific weights. Its authors supply training/evaluation code and
   pretrained weights, and document the synthetic-to-real generalization issue:
   [official repository](https://github.com/IDKiro/DehazeFormer).
   A stronger architecture is a candidate improvement, not a guaranteed gain here.
4. If collecting representative paired training data is feasible, fine-tune
   pretrained weights using scene-disjoint validation. Compare L1/Charbonnier
   reconstruction loss with an added SSIM term; choose weights and checkpoints
   by validation metrics. Use paired crops/flips/rotations and realistic haze
   diversity. This is a proposed experiment; no training was performed in this run.
5. Evaluate real captured haze as well as synthetic haze. The DehazeFormer
   authors recommend mixed data for real-world use. Avoid selecting by PSNR
   alone: retain SSIM, visual inspection, and runtime/memory requirements.

There is no universally achievable maximum score. PSNR and SSIM depend on the
dataset, preprocessing, and evaluation protocol. SSIM 0.90 is not 90% accuracy.
