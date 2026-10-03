# Production model assets

`scripts/finalize_release.py` creates exactly two runtime files here:

- `segformer_b1_detail_768_v3.onnx`
- `release.json`

The ONNX file is intentionally not included in the source archive. Track it
with Git LFS or attach it to a versioned GitHub Release. The release JSON pins
its SHA-256, calibration choice, dataset fingerprint, best checkpoint, ONNX
parity results, and frozen holdout report.
