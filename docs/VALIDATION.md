# v2 integration validation

This release changes packaging, explicit asset lookup and headless/review
orchestration. It does not retrain, recalibrate or repeat holdout evaluation.
The composition workspace records detailed validation in `VALIDATION.md` and the
[qualification receipt](../provenance/INTEGRATION_VALIDATION.json); runtime assets/sources are authenticated
by `tools/verify_release.py`.

Run `python -m pytest tests -q` after installing the test dependencies. New tests
cover headless pending-review behavior, output path containment, authenticated
resume and artifact tampering. `paper-stage1 verify --models models` checks the
frozen model pair without opening a GUI.

All 11 Stage 1 tests passed against installed wheels on Windows and Ubuntu/WSL.
Fresh four-stage CPU runs on both platforms produced a supported connected
four-piece assembly. Container deployment remains a separate qualification.
Generated capture checks establish interchange,
not real-photo reconstruction accuracy or held-out segmentation metrics.
