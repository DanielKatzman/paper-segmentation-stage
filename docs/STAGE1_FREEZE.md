# Stage 1 freeze record

## Approved training run

- Architecture: SegFormer-B1 Detail
- Input: 768 × 768 letterboxed RGB, ImageNet normalization
- Run: `segformer_b1_detail_768_real_v3`
- Frozen dataset fingerprint:
  `225061bdeab36f7a10d5b9080c01ad013db0cb3b798d53e6ddfddcd86834ffaf`
- Completed epochs: 8
- Best checkpoint: human epoch 8 (zero-based epoch 7)
- Best validation score: `0.9259111612330179`
- Best validation median IoU: `0.9948151885478516`
- Best validation boundary F1@2: `0.9862764854614066`
- Validation catastrophic rate: `0.0625` (2/32)

The two recurrent validation failures were false-positive distractor cases:
`real_715e6af5ebcc37c1` and `real_af67cab151dee196`. The release workflow
therefore calibrates both threshold and an explicit `largest_non_border`
target-component policy on calibration only. Holdout remains untouched until
that choice is locked.

## Product decisions

- SegFormer v3 is the only production and annotation model.
- Scanner and phone images share one automatic inference path.
- The old deterministic guesser is removed.
- The annotation GUI predicts in one CPU background worker and caches by source
  hash plus complete model/release policy identity.
- The batch stage can be unattended, review only warnings, or review every
  result. Manual correction is the fallback.
- Stage 2 communicates only through the versioned JSONL/artifact contract.
- The historical research repository is not modified by this cleanup.

This record describes the checkpoint before ONNX calibration and holdout. The
final `models/release.json` becomes the authoritative deployment record after
those gates pass.

The complete per-epoch and per-image source report is retained at
`docs/reports/training_report_v3.json`.
