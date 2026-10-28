# Stage 1 v2 orchestrator contract

Distribution version 2.0.0 packages core software 1.0.0 and frozen SegFormer v3.
Model identity is independent of the package/interface version. Resolve the
compatibility manifest and pin exact source/model/configuration hashes at build
time. The model assets remain external to the wheel; provide `--models` explicitly.

Use Python 3.14 with CPU ONNX Runtime and the release constraints. Verify source
inventories with `tools/verify_release.py` and installed assets with
`paper-stage1 verify --models <models>`. Do not import generators or training code
to run the stage.

Run `paper-stage1 run --input <captures> --output <job/stage1_native> --models
<models> --capture-profile <profile.json>`. Original images and rendered PDF pieces
retain their native capture pixel frame. PDF rendering defaults to 300 DPI;
declare a consistent acquisition and do not independently rescale pieces.

The optional profile is an explicit declaration, copied into the output and
bound to the request. It must use `stage1.capture-profile.v1`, shared native pixel
scale, relative scale 1, rotation/translation, no per-piece resizing and no
perspective. Raw inference may declare `segformer_v3_predicted` or
`predicted_reviewed`, never generator-exact masks. The segmenter cannot verify
camera calibration or infer that the capture assumptions are true.

The canonical handoff remains `segmentation_manifest.jsonl` with schema
`paper-segmentation-manifest-1.0`, source hashes, stable IDs, acceptance/review
state, tight RGBA crops, full and cropped binary masks and bbox translation.
`integration_request.json` binds source files, model/configuration, acquisition
profile and interface identity. `integration_receipt.json` seals the published
artifacts. These files add provenance; they do not replace the canonical handoff.

Exit 0 means a complete accepted batch. Exit 3 means review is required; stop
full-page reconstruction and expose pending pieces. Exit 2 means failure and
130 interruption. Progress uses stderr; stdout ends in one JSON object. The
command never opens a GUI. `--review flagged` is the default, `all` requires every
piece to be reviewed, and `none` permits automatic structural acceptance. No
headless path pretends a human approved a piece.

Resume uses the identical command plus `--resume`. Changed source/configuration
needs a new job. Completed publications are reused only after checksum checks;
interrupted jobs regenerate final rows using their authenticated prediction cache.
A stale lock must be handled by an operator after confirming the recorded PID is
stopped. For manual correction, use the desktop review workflow and import a
new, independently checked Stage 1 handoff into a new downstream job.

Composition must retain the raw output and apply
`bounded-predicted-mask-native-frame.v2` into a separate directory before Stage 2.
That adapter records bounded cleanup, crop margins and coordinate offsets without
resizing. Pieces touching the capture frame or exceeding cleanup bounds require
review. Never relabel adjusted predicted masks as exact masks. The composition
repository owns the adapter and capture/review UI; Stage 1 owns segmentation.
