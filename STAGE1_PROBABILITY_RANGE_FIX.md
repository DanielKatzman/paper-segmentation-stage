# Stage 1 probability-range hotfix

This patch fixes a production inference crash in which ONNX Runtime returned a
sigmoid endpoint a few float32 ULPs outside `[0, 1]`. The strict public result
contract correctly rejected that value, but the runtime had not normalized
finite-precision endpoint noise first.

The patch:

- clips only tiny ONNX probability excursions (tolerance `1e-5`);
- continues to reject NaN, infinity, and materially invalid outputs;
- clips the restored original-resolution probability map as a second numerical
  boundary;
- keeps `SegmentationResult.validate()` strict and unchanged;
- adds regression tests for accepted roundoff and rejected invalid output.

No model, threshold, calibration, mask semantics, or manifest schema changes.
The ONNX file does not need to be re-exported.

## Apply

Extract the update ZIP into the repository root and allow it to merge/replace
the included files. Then run:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

Resume the interrupted command with the same input and output folders:

```powershell
python .\scripts\segment_folder.py `
  "C:\Users\danka\Desktop\paper-segmentation-stage\real_puzzle_captures" `
  "C:\Users\danka\Desktop\paper-segmentation-stage\stage1_real_puzzle_output" `
  --review all
```

Do not delete the output folder and do not add `--reprocess`. The first two
completed rows will be reused and processing will continue at the third image.

## Commit and tag

After the six-piece run succeeds:

```powershell
git add src/universal_segmenter/inference/cpu_runtime.py `
        src/universal_segmenter/inference/preprocess.py `
        tests/test_probability_contract.py `
        STAGE1_PROBABILITY_RANGE_FIX.md
git commit -m "Clamp numerical probability overflow in ONNX inference"
git push
git tag -a v1.0.1 -m "Stage 1 probability range hotfix"
git push origin v1.0.1
```
