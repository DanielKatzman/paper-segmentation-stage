# Paper Segmentation Stage

This is the frozen first stage of the torn-paper reconstruction pipeline. It
finds one target paper piece in either a scanner capture or a phone photograph,
without asking the user to select a capture mode. The production network is
only **SegFormer-B1 Detail v3 at 768 px**. ResNet and the old deterministic
guesser are deliberately not part of this repository.

The source repository does not contain the large ONNX binary. The finalization
workflow below creates the two files that define the production release:

```text
models/
  segformer_b1_detail_768_v3.onnx
  release.json
```

`release.json` pins the ONNX SHA-256, frozen-dataset fingerprint, checkpoint,
parity check, calibration decision, and untouched-holdout report. The GUI and
batch pipeline refuse to use an unverified release.

## Architecture

| Module | Owns | Depends on |
|---|---|---|
| `inference` | letterbox, ONNX CPU inference, calibrated threshold, target-component policy | `models/` release files |
| `annotation` | recursive folder scan, background cache, split assignment, manual correction | `inference` |
| `pipeline` | production batch output and stable JSONL contract | `inference`, correction editor |
| `generation` | real/synthetic dataset compilation | three generator input folders |
| `training` | SegFormer training, best/last checkpoints, exact resume | frozen dataset |
| `evaluation` | calibration-only selection and one-time holdout gate | exported ONNX, frozen dataset |

The next reconstruction stage must read `segmentation_manifest.jsonl` and the
artifact paths it names. It must not import training, generator, or annotator
internals. That keeps Stage 1 replaceable.

## Exact finalization steps on your Windows PC

Start in this new repository, not the old research folder.

1. Copy the completed assets from the old project into the same relative paths:

```text
old project/data/complete_dataset_v3
    -> this project/data/complete_dataset_v3

old project/experiments/segformer_b1_detail_768_real_v3/checkpoints/best.pt
    -> this project/experiments/segformer_b1_detail_768_real_v3/checkpoints/best.pt
```

Only `best.pt` is required to make the production model. Keep `last.pt` and the
training report in your private training archive if you want full provenance;
they are not runtime assets.

2. Reuse the CUDA/PyTorch environment that already trained v3. This avoids
replacing your known-good GTX 1070 PyTorch build:

```powershell
cd C:\Users\danka\Desktop\paper-segmentation-stage
$PYTHON = "C:\Users\danka\Desktop\universal-paper-segmenter\.venv\Scripts\python.exe"
& $PYTHON -m pip install -r requirements-training.txt
```

3. Export the exact approved checkpoint and verify PyTorch/ONNX parity:

```powershell
& $PYTHON scripts\export_segformer.py
```

This script refuses the wrong run, wrong dataset fingerprint, wrong best epoch,
or wrong validation score. The expected checkpoint is human epoch 8 with score
`0.9259111612330179`.

4. Select threshold and postprocessing using **calibration only**:

```powershell
& $PYTHON scripts\calibrate_release.py
```

5. Run the untouched holdout exactly once:

```powershell
& $PYTHON scripts\evaluate_release.py
```

Inspect both:

```text
exports/holdout_evaluation.json
exports/holdout_preview.jpg
```

Continue only if the JSON says `"valid": true` and the preview looks correct.
If it fails, do not tune on holdout. Add training/calibration data, create a new
dataset and run name, retrain, and reserve a new holdout.

6. Create the immutable production model pair:

```powershell
& $PYTHON scripts\finalize_release.py
```

7. Smoke-test the production batch stage on a separate folder:

```powershell
& $PYTHON scripts\segment_folder.py `
  C:\path\to\captures C:\path\to\stage1_output --review flagged
```

Review modes are:

- `flagged` (default): open the editor only for structural warnings;
- `all`: manually approve every mask for maximum safety;
- `none`: unattended processing; warnings remain in the manifest.

The command is resumable. Existing outputs are reused only when source hash,
model hash, threshold, policy, and expected files still match.

8. Launch the source version of the annotation tool:

```powershell
& $PYTHON scripts\run_annotator.py
```

9. Build the final Windows application and the Apple-Silicon build kit:

```powershell
& $PYTHON -m pip install -r requirements-portable-build.txt
& $PYTHON scripts\export_standalone_annotator.py --all
```

Outputs:

```text
release/PaperSegmentationAnnotator-Windows-x86_64.zip
release/PaperSegmentationAnnotator-macOS-arm64-build-kit.zip
```

PyInstaller cannot build a macOS app on Windows. Send the build-kit ZIP to the
Apple-Silicon teammate. They extract it and double-click
`BUILD_MAC_APP.command`; it creates the native macOS application ZIP.

## Production output contract

Each batch output folder contains:

```text
stage1_output/
  masks/                         full-resolution binary masks
  pieces/                        tight RGBA paper crops
  piece_masks/                   crop-coordinate binary masks
  overlays/                      human-review previews
  cache/                         model-hash-keyed reusable predictions
  segmentation_manifest.jsonl    canonical interface to Stage 2
  segmentation_run.json          run summary
```

Every manifest row contains source SHA-256, original size, bounding box,
artifact paths, model/release identity, threshold, postprocessing policy,
automatic/manual decision source, acceptance state, reason codes, and
diagnostics. See `schemas/segmentation_manifest.schema.json`.

## Collecting and training a future release

The generator scans all three inputs itself:

```text
data/generator_inputs/
  backgrounds/
  pdfs/
  real_pieces/
    images/
    masks/
    manifests/
```

Copy the annotation workspace's images, masks, and manifests together so
train/validation/calibration/holdout assignments are preserved. Then run:

```powershell
& $PYTHON scripts\generate_dataset.py
& $PYTHON scripts\freeze_dataset.py
& $PYTHON scripts\train_segformer.py
```

Training writes `last.pt` frequently and `best.pt` only after a completed
validation epoch. Ctrl+C once or `scripts/request_stop.py` stops safely; rerun
the training script to resume the exact cursor. A future release must use a new
dataset folder, run name, release identifier, and untouched holdout.

## Repository policy

- Commit source, schemas, small reports, and `release.json`.
- Store the ONNX with Git LFS or a versioned GitHub Release.
- Do not commit raw captures, generated datasets, checkpoints, caches, build
  folders, or teammate annotation workspaces.
- Keep the old research repository as an archive until this release passes its
  holdout and standalone smoke tests.
