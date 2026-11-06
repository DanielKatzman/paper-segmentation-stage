# Paper Segmentation Stage — v2.5.0

Stage 1 of the paper-reconstruction pipeline segments paper fragments from photos
and scans. One image can produce multiple independently identified pieces, each
with its own native-resolution mask, transparent crop and review state.

The model is **SegFormer-B1 Detail v3**, unchanged in v2.5.0. Inference runs on
CPU using a 768 × 768 letterboxed view; masks and crops are restored to the
original image coordinates. Training data, PyTorch and a GPU are not required
to run the released tool or headless inference.

[Download v2.5.0](https://github.com/DanielKatzman/paper-segmentation-stage/releases/tag/v2.5.0)
· [Teammate guide](docs/TEAMMATE_COLLECTION_GUIDE.md)
· [Dataset owner quickstart](docs/MULTI_PIECE_QUICKSTART.md)
· [Orchestrator handoff](docs/ORCHESTRATOR_HANDOFF_V2_5.md)

## What v2.5.0 supports

- **Separated, fully visible fragments:** one or many pieces per photo, with
  independent correction, acceptance and rejection. Touching/overlapping pieces
  and recovering occluded paper are outside the current capture workflow.
- **Training captures with zero, one or many pieces:** arbitrary fragments from
  unrelated or unknown pages are valid. Page IDs are optional; collecting an
  entire original page is not required.
- **A portable annotation tool:** automatic suggestions for untouched captures,
  progress and timing, cancellation, manual edits, split/merge operations and
  revision history.
- **Dataset preparation:** checked collection ZIPs, native integer instance
  labels, union masks, raster outlines, central split assignment and immutable
  dataset versions. Existing datasets are kept separately.
- **Synthetic scenes:** procedural paper fragments, reviewed real cutouts,
  retextured shapes, backgrounds, capture effects and empty scenes. Generation
  is configured, seeded and resumable.

The training/data foundation is included, but this release does not contain a
newly trained multi-piece model. Review real predictions before accepting them.

## Download the tool

| Asset | Contents |
| --- | --- |
| [Windows tool](https://github.com/DanielKatzman/paper-segmentation-stage/releases/download/v2.5.0/PaperSegmentationAnnotator-Windows-x86_64.zip) | Standalone annotator with the CPU model and runtime. Extract the whole ZIP before running the executable. |
| [macOS Apple-Silicon build kit](https://github.com/DanielKatzman/paper-segmentation-stage/releases/download/v2.5.0/PaperSegmentationAnnotator-macOS-arm64-build-kit.zip) | Source and model bundle to build on a native arm64 Mac. This is a build kit, not a prebuilt `.app`. |
| [Python wheel](https://github.com/DanielKatzman/paper-segmentation-stage/releases/download/v2.5.0/paper_segmentation_stage-2.5.0-py3-none-any.whl) and [models](https://github.com/DanielKatzman/paper-segmentation-stage/releases/download/v2.5.0/paper-segmentation-stage-v2.5.0-models.zip) | Python/headless installation. Install the wheel with its `runtime` extra and provide the extracted `models/` directory explicitly. |

The [release page](https://github.com/DanielKatzman/paper-segmentation-stage/releases/tag/v2.5.0)
also contains the source archive, handoff, validation report, asset manifest and
`SHA256SUMS.txt`. The Windows tool bundles its model; the Python wheel does not.

## Collect real training images

1. Open the annotator and choose **Open / create workspace**. Keep the workspace
   outside the input photo folder and application folder; reopen it to resume.
2. Choose the capture type and add a photo/scan folder. Contributor, collection,
   known page ID and session are optional metadata. Preserve known repeat-piece
   or session identities so the dataset owner can group related observations.
3. Select a capture. Automatic suggestions run for untouched images. Review each
   proposal separately, correct its mask, reject false detections and add missed
   pieces. The model's proposals are not manual approvals.
4. When every visible piece in that image is labeled and accepted, choose
   **Confirm every VISIBLE piece labeled**. This concerns the current image,
   regardless of whether its fragments form a complete page.
5. For a true negative, reject false proposals and choose **Confirm NO pieces**.
   Rejection alone does not confirm an empty image. Mark a negative as a
   **Clean compositing background** only when it is suitable for synthetic use.
6. Choose **Export reviewed collection** and send that ZIP to the dataset owner.
   The default export includes reviewed/confirmed-empty images and omits
   unfinished or excluded captures.

Collections contain native images, instance labels, union masks, outlines,
metadata, checksums and a preview sheet. The owner-side `collection-assets`
command extracts training-eligible real pieces/backgrounds for the generator.

**Publish approved pieces** is a separate export of accepted reconstruction
observations and their crops/masks. It does not certify a complete physical
puzzle inventory. The [teammate guide](docs/TEAMMATE_COLLECTION_GUIDE.md) explains
editing controls, empty images, saved revisions and the two exports.

## Run from source

Python 3.12 is used for the portable GUI build. The headless CPU runtime and wheel
have also been checked on Python 3.14. Use a separate environment for training.

```text
git clone https://github.com/DanielKatzman/paper-segmentation-stage.git
cd paper-segmentation-stage
git lfs install
git lfs pull
python -m venv .venv
```

Activate the new environment (`.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on Linux/macOS), then run:

```text
python -m pip install -e ".[runtime,data]"
python tools/verify_release.py
python scripts/run_annotator.py
```

Source GUI use requires a working Tk installation. Its launcher finds the model
under this checkout's `models/` directory. The packaged Windows app needs neither
Python nor Tk installed separately. For a wheel-only/headless installation, use
`paper-stage1` with `--models /absolute/path/to/models`.

`constraints.txt` records the pinned Python 3.14 CPU integration dependencies.
`tools/setup.py` creates that runtime environment, requires Python 3.14 and
refuses to overwrite an existing environment. It is not the full training/test
setup. Training dependencies and hardware planning are covered in the
[training status](docs/P40_TRAINING_PLAN_AND_STATUS.md) and
[university setup](docs/UNIVERSITY_GPU_SETUP.md).

The two verified model files are:

```text
models/
  segformer_b1_detail_768_v3.onnx
  release.json
```

`release.json` binds the model hash, calibration and historical v3 qualification.
The new multi-piece path preserves the calibrated threshold and extracts all
qualifying connected components from the probability map. The old
single-target component-selection policy remains available in legacy mode.

## Headless segmentation

```text
paper-stage1 verify --models models
paper-stage1 run --input captures --output new_stage1_output --models models --piece-mode multi --review all
paper-stage1 inspect --input new_stage1_output
```

`python -m universal_segmenter.release_cli` is equivalent to `paper-stage1`.
This interface is headless: it never opens a correction window. The default
piece mode is `multi`; `--piece-mode single` selects the historical one-target
pipeline. The orchestrator also supplies its bound acquisition declaration with
`--capture-profile /job/capture_profile.json`.

| Review setting | Behavior |
| --- | --- |
| `--review all` | Every candidate remains pending individual review. |
| `--review flagged` (default) | Structurally flagged candidates remain pending; other candidates can be automatically accepted. |
| `--review none` | No requested manual-review pass; structural failures still remain unaccepted. |

Exit codes are `0` ready, `3` review required, `2` failed and `130` interrupted.
Progress is written to stderr; stdout contains one JSON result. A zero-proposal
prediction produces an explicit pending empty-source placeholder; it does not
automatically certify that the image contains no paper.

Use a new output folder for a changed request. `--resume` requires identical
inputs, mode, configuration, model, package source and runtime dependencies;
modified published artifacts are rejected. Begin new jobs when migrating from
an older Stage 1 version.

For interactive folder review, use `paper-segment` or
`python scripts/segment_folder.py INPUT OUTPUT --review flagged`. The collection
app is launched with `python scripts/run_annotator.py`; its old single-piece UI
is available with `--legacy`.

## Output contract and Stage 2

```text
stage1_output/
  masks/                         full native-size per-piece binary masks
  pieces/                        tight native RGBA crops
  piece_masks/                   crop-coordinate binary masks
  overlays/                      review previews
  cache/                         checksummed prediction caches
  segmentation_manifest.jsonl    one row per observation
  segmentation_run.json          source inventory and run summary
  integration_request.json       headless request identity
  integration_receipt.json       headless publication checksums/result
```

Each piece observation has a unique `capture_id` and `artifact_id`. Pieces from the same
photo share `source_capture_id`, `source_sha256` and `source_path`. Each row names
its own mask/crop files and records native size, crop offset, model identity,
acceptance and review state. `source_sha256` is not a unique piece/artifact key.

The schema remains `paper-segmentation-manifest-1.0`, with additive source-linkage
fields. The number of rows can exceed the number of photos. Empty-source
placeholders have `empty_proposal: true` and are unaccepted. The run summary
separates `source_count` from `piece_count`; `complete` means all discovered
sources were processed, not that all pieces of an original page were supplied.

**Stage 2 needs no input-reader change for this format.** Its reader consumes
individual accepted manifest rows and their crop/mask paths, uses unique
`capture_id` values, and supports multiple rows sharing one source photo. The
reader in Stage 2 `main` (`84c36be`) is byte-identical to the orchestrator-pinned
version (`8b09586`). Its per-piece output/cache identity also uses the individual
capture or record fingerprint, rather than deduplicating by source-image hash.
See the [Stage 2 reader](https://github.com/DanielKatzman/paper-reconstruction-stage2/blob/8b095867f1eea7216920196fd36c08a4f57b2820/src/paper_reconstruction_stage2/production.py)
and [compatibility check](provenance/STAGE2_V2_5_INPUT_COMPATIBILITY.json).

Keep the orchestrator's existing native boundary adapter before Stage 2: it
checks connectivity and adds the crop margin used by downstream native geometry.
The **orchestrator still needs its GUI, review publication and piece-inventory
migration**. Its previous one-photo/one-piece assumptions are separate from
Stage 2's input reader. The [v2.5 handoff](docs/ORCHESTRATOR_HANDOFF_V2_5.md) lists
those changes and the checks required before updating its production pin/image.

For reconstruction, preserve a common physical pixel scale across captures:
consistent scan DPI, or fixed phone height, zoom, output resolution and a parallel
paper plane. Stage 1 does not resize individual exported pieces, correct
perspective, determine page ownership or deduplicate the same physical fragment
across different photos.

## Build datasets and generate synthetic scenes

```text
python scripts/paper_scenes.py --help
python scripts/paper_scenes.py import teammate_collection.zip data/multi_piece
```

Use the [owner quickstart](docs/MULTI_PIECE_QUICKSTART.md) for the full sequence:
import collections, resolve conflicting annotations, assign central grouped
splits, register training-eligible assets, generate synthetic scenes and freeze
a new dataset. `configs/multi_piece_generator_v1.json` controls source/background
mixtures, piece-count bins, spacing, effects, resolution and seed.

The default generator includes empty scenes and count bins up to 17–32 pieces;
these are configurable generation ranges, not a guaranteed detection capacity.
Only centrally assigned training assets are eligible for synthesis. Real masks
supply cutouts that can be placed on varied backgrounds alongside procedural or
retextured pieces.

Keep `complete_dataset_v3` unchanged and use new folders under `data/multi_piece/`.
Adding old samples requires explicit auditing that every visible real piece is
labeled; old one-target masks can leave other paper pieces unlabeled. Preserve
historical evaluation groups and keep related repeat captures in one split.
The builder refuses overwriting a dataset and records a content-hash freeze.

## Validation and maintenance

The v2.5 release was checked with 51 tests, own-application Tk controls, a
Python 3.14 CPU/wheel installation and the extracted Windows executable. Seven
real 12-megapixel photos produced 28 separate pieces: the native boundary adapter
preserved all 28 with zero pixels removed, and the Stage 2 input loader accepted
all 28 unique IDs. These are integration controls, not a general accuracy score
or a newly qualified four-stage GUI/Docker deployment.

- [Release validation](provenance/RELEASE_V2_5_VALIDATION.json)
- [Validation scope](docs/VALIDATION.md)
- [Model provenance and freeze record](docs/STAGE1_FREEZE.md)
- [Release notes](docs/RELEASE_V2_5.md)

To build the tool in a separate source/build environment:

```text
python -m pip install -r requirements-portable-build.txt
python scripts/export_standalone_annotator.py --all
```

This produces the Windows ZIP and macOS source build kit on Windows. The native
macOS application must be built and checked on an Apple-Silicon Mac.

| Package area | Responsibility |
| --- | --- |
| `inference` | Verified v3 model, letterboxing, CPU probabilities and native restoration. |
| `scenes` | Multi-piece annotation, collection exchange, synthetic scenes and frozen datasets. |
| `pipeline` | Native per-piece artifacts and batch publication. |
| `annotation` | Mask editor and legacy single-piece interface. |
| `training`, `losses` | Dataset loading, sampling, losses, checkpoints and resume. |
| `evaluation` | Model calibration, metrics and release evaluation. |

Commit source, schemas, small reports and release metadata. Keep raw captures,
annotation workspaces, generated datasets, checkpoints, caches and build outputs
outside Git; the ONNX payload is versioned with Git LFS and the release assets.
