# Distribution 2.0.0

Integration packaging for Stage 3 v2.0.0 and Stage 4 v2.0.5. Core software remains
1.0.0 and the production model remains SegFormer v3 with unchanged weights,
threshold and inference policies. No new holdout evaluation was performed.

- Add `paper-stage1 run/inspect/verify` with explicit models, JSON results and
  stderr progress.
- Add headless pending-review behavior and checksum-bound publication/resume.
- Add CPU setup, constraints, compatibility metadata and source/model verification.
- Preserve exact release bytes across platforms and document boundary-adapter
  ownership and acquisition declarations.
- Preserve the interactive desktop command and canonical segmentation contract.

This is a local release candidate until its source/model assets and validation
receipts are published together under an immutable release tag.
