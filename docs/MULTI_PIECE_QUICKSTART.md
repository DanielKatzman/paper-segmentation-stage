# Multi-piece foundation: owner workflow

Version 2.5.0 adds collection/generation/data tools and a per-piece headless publication around the unchanged SegFormer v3 model. Keep `complete_dataset_v3` and the existing production checkout intact. Training follows collection. The orchestrator must apply [the v2.5 migration handoff](ORCHESTRATOR_HANDOFF_V2_5.md) before adopting this version; a new four-stage production image is not included here.

Training captures may contain zero, one or many arbitrary fragments, including pieces from unrelated or unknown pages. A complete page inventory is never required. Page IDs are optional grouping information when known. The app automatically proposes pieces on selection; review every visible piece before marking that individual image complete. Collection exports include native instance/union masks and raster outlines; `collection-assets` derives the real cutouts and clean backgrounds used by the synthetic generator.

## Start the collection app

Give teammates `release/PaperSegmentationAnnotator-Windows-x86_64.zip` and the [collection guide](TEAMMATE_COLLECTION_GUIDE.md). They extract the whole ZIP and run its executable. Source launch:

```powershell
python scripts/run_annotator.py
# Existing single-piece UI remains available:
python scripts/run_annotator.py --legacy
```

The source app needs a working Python Tk installation and runtime dependencies. `--workspace PATH` reopens a collection. The packaged app bundles these requirements. The portable GUI was built with Python 3.12 because this host's Python 3.14 install lacks Tcl data; headless runtime controls are checked separately on Python 3.14.

## Install command dependencies

Use a separate environment, preserving the known working CUDA training environment:

```powershell
python -m venv .data-venv
.\.data-venv\Scripts\python.exe -m pip install -e ".[runtime,data]"
```

Commands below use `python scripts/paper_scenes.py`; substitute that environment's Python. Installed users can run the equivalent `paper-scenes` command. `python scripts/paper_scenes.py --help` lists all commands.

## Import teammate data

```powershell
python scripts/paper_scenes.py import teammate_collection.zip data/multi_piece
```

Imported packages remain under `data/multi_piece/collections/<collection-id>/`. `capture_index.json` identifies the authoritative first annotation for a source; `import_reports/` lists duplicates and conflicts. Reimport is idempotent. Conflicting versions are retained separately, and a dataset build refuses competing labels for the same capture. Reconcile them in an annotation workspace and deliberately choose the corrected revision; never merge by overwriting files.

After inspecting a corrected package, select its preserved revision explicitly with `paper-scenes resolve-annotation data/multi_piece SOURCE_CAPTURE_ID --collection COLLECTION_ID`. The selection is audited and leaves earlier packages/frozen datasets intact. `build-dataset --central data/multi_piece` uses these authoritative selections; explicit `--source` roots instead include every eligible scene in those roots and refuse conflicts. Avoid selecting both competing roots.

To correct an imported collection, run `paper-scenes clone-workspace COLLECTION_DIRECTORY NEW_WORKSPACE` and open the new workspace in the annotator. Checked packages and frozen datasets reject edits. Export the corrected workspace as another package, import it, inspect the conflict report and explicitly select the intended revision.

The collection's `source_name` is provenance only; training inputs use included native decoded images and relative paths. Identical normalized pixels deduplicate even after a file rename. Different photos of the same physical piece still need shared page/physical identity for split grouping.

## Assign central splits

Teammates do not independently assign dataset percentages. Register complete collection scenes centrally:

```powershell
python scripts/paper_scenes.py assign-splits `
  --source data/multi_piece/collections/COLLECTION_ID `
  --ledger data/multi_piece/source_inventory/splits.json `
  --legacy C:/Users/danka/Desktop/universal-paper-segmenter/data/complete_dataset_v3
```

Repeat `--source` for several collections. Shared source, page, physical-piece or session tokens form one group. Existing assignments are retained; changed ratios need a new ledger. If new data bridges conflicting protected groups, quarantine/reconcile it. New holdout pages/sessions should be independent of historical data. Small pilots may not populate all four splits; reports explicitly show `evaluation_ready: false` until real train/validation/calibration/holdout coverage exists.

## Generate multi-piece synthetic data

Discover existing background/PDF/real-cutout sources into a NEW eligibility inventory:

```powershell
python scripts/paper_scenes.py inventory `
  C:/Users/danka/Desktop/universal-paper-segmenter/data/generator_inputs `
  data/multi_piece/source_inventory/assets.json
```

PDF/background eligibility initially says `unassigned`; review `assets.json` and set eligible training assets to `"split": "train"`. Real cutouts retain their existing source assignments. `--declare-training-assets` explicitly declares all discovered PDFs/backgrounds training-only; use it only for a collection already checked against protected sources. Never mark holdout pieces/textures/backgrounds train to make generation pass.

Add new reviewed training cutouts and explicitly clean backgrounds after central split assignment:

```powershell
python scripts/paper_scenes.py collection-assets `
  data/multi_piece/collections/COLLECTION_ID `
  data/multi_piece/source_inventory/assets.json `
  --ledger data/multi_piece/source_inventory/splits.json
```

Edit a copy of `configs/multi_piece_generator_v1.json`. It controls count bins (including zero), master/training resolution, source mixtures, profile mixtures, minimum gap, retry limits, effects and `cells`/`scatter`/`mixed` layout. If no real cutouts exist yet, use `"source_weights": {"procedural": 1}`. If only phone backgrounds exist, use `"profile_weights": {"phone": 1}` until scanner backgrounds are collected. Unsupported source mixtures fail explicitly.

```powershell
python scripts/paper_scenes.py generate `
  --inventory data/multi_piece/source_inventory/assets.json `
  --config configs/multi_piece_generator_v1.json `
  --output data/multi_piece/synthetic_scenes_v1
```

Use `--resume` only for an identical request. Changed configuration, inventory or source bytes require a new output version. Inspect `generation_report.json` and `previews/contact_sheet.jpg`. Requested piece counts are exact; impossible placement fails instead of reducing the count. The report flags pieces/gaps lost at the proposed training resolution. Master truth is preserved for later tiling or higher-resolution views.

Source recipes record training eligibility and image/PDF/mask hashes. All generated training examples remain train-only. These segmentation augmentations are not calibrated Stage 3/4 reconstruction truth.

## Freeze a new dataset

```powershell
python scripts/paper_scenes.py build-dataset `
  --source data/multi_piece/collections/COLLECTION_ID `
  --source data/multi_piece/synthetic_scenes_v1 `
  --ledger data/multi_piece/source_inventory/splits.json `
  --legacy C:/Users/danka/Desktop/universal-paper-segmenter/data/complete_dataset_v3 `
  --output data/multi_piece/datasets/multi_piece_v1

python scripts/paper_scenes.py verify-dataset data/multi_piece/datasets/multi_piece_v1
```

By default, `--legacy` protects historical evaluation groups but does not blindly add old samples. To combine audited legacy TRAIN examples, supply `--legacy-exhaustive-ids audited_ids.json`, a JSON list of sample IDs whose masks exhaustively label valid pieces. The builder never imports legacy validation/calibration/holdout into training. Photos with paper distractors left unlabeled under the old one-target task need correction or exclusion.

The new dataset has `samples.csv`, native images/instance labels/union masks, source/split receipts, coverage report and `scene_dataset_freeze.json` with content hashes. It refuses overwrite and detects changed artifacts. A structurally valid small dataset can be frozen for inspection; the training freeze gate additionally requires real data in all four splits. The existing `PaperDataset` reads both formats, includes instance labels for metrics, and handles true empty masks with finite distance/loss targets.

`TrainingConfig.scene_count_weights` optionally balances `0`, `1`, `2-4`, `5-8`, `9-16` and `17+` scene bins within the existing sample-kind mixture. Real capture draws also balance available phone/scanner profiles. Leave it empty to retain the legacy draw policy; enabled weights must cover the available bins in every positively weighted category. Validation records matched-instance precision/recall and exact counts in addition to binary and boundary metrics.

`scenes.dataset.load_view` provides aligned letterboxed views and explicit native tile coordinates without discarding partial-piece labels at internal tile edges. No final GPU/batch/resolution policy is selected by the foundation.

## Approved reconstruction pieces

```powershell
python scripts/paper_scenes.py publish-pieces WORKSPACE NEW_OUTPUT_FOLDER
```

One row per approved observation, with unique artifact filenames and shared `source_capture_id`, is written in the existing piece-level schema. Rejected proposals stay in revision history. `physical_inventory_confirmed` remains false: source/candidate/physical inventory migration in the orchestrator is a separate milestone. The exported pieces have been exercised through its existing connectivity/crop-margin adapter.

Generated ground truth is ready for training, but is not a human review decision. Normal piece publication requires individual manual acceptance even for a generated example opened in the app. The separate `--include-generated-fixtures` option is for adapter diagnostics and preserves `synthetic_truth` provenance; it never labels generated truth as human-reviewed.

## Controls and example data

```powershell
python scripts/create_scene_demo.py NEW_DEMO_FOLDER
python scripts/run_annotator.py --self-check exports/runtime_control.json
python -m pytest -q
python tools/verify_release.py
```

The headless self-check exercises the actual ONNX, four synthetic fixture pieces, a true negative, portable collection round-trip, repeated import, dataset freeze and piece publication. It does not claim real-image accuracy. `examples/multi_piece_demo` is a generated example collection; never treat it as a real holdout.

The GUI package can run the same check with `PaperSegmentationAnnotator.exe --self-check ABSOLUTE_REPORT_PATH`. Package builds exclude Torch/Transformers and bundle the unchanged verified model. The macOS build kit requires native build/qualification on a Mac.
