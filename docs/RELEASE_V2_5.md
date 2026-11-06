# v2.5.0 — multi-piece segmentation and collection

The same SegFormer-B1 Detail v3 model now supplies separate proposals for multiple
separated fragments in one photo. No model retraining is included in this release.

- Portable Windows annotation app: automatic suggestions, progress/timing,
  cancellation, independent correction/approval/rejection and saved revisions.
- Arbitrary fragments and optional page IDs; zero, one and many pieces per
  training photo. Native instance masks, union masks, raster outlines and checked
  collection ZIPs. Existing datasets remain separate and unchanged.
- Synthetic scenes with procedural/real/retextured pieces, varied backgrounds,
  count mixtures including negatives, deterministic generation/resume and exact
  instance labels. Central grouped splits and immutable dataset assembly.
- Headless multi-piece default: unique per-piece IDs/artifacts, source inventory,
  native transparent crops, explicit empty-prediction review, verified resume
  and mask/alpha/overlap inspection. Historical `--piece-mode single` retained.
- Source/wheel installation uses explicit CPU models; portable GUI bundles its
  model/runtime. The Apple-Silicon macOS asset is a **source build kit**, requiring
  a native Mac to build; it is not a prebuilt `.app`.

The orchestrator's GUI/cardinality migration is required before adopting this
version. Forward [the migration handoff](ORCHESTRATOR_HANDOFF_V2_5.md) to its owner.
Native downstream per-piece mask/crop geometry and the existing boundary adapter
are retained. A new four-stage Docker deployment is not part of this release.

Continue reviewing real predictions. Correct counts on the local real-photo
controls are useful evidence, not a general segmentation accuracy benchmark.
