# Stage 1 v2.5.0: orchestrator migration handoff

Release: `DanielKatzman/paper-segmentation-stage`, tag `v2.5.0`.
The model is the unchanged SegFormer-B1 Detail v3 at 768 px. Its ONNX SHA-256 is
`cbd43f3f305d44c811d70f1ab604c7dbef9d5ddb2eecd9fd0dbe3f52e3345417`.

## Scope and current compatibility

Stage 1 now accepts photos with multiple **separated, fully visible** fragments.
The headless default is multi-piece; `--piece-mode single` retains the historical
single-target path. The collection app also supports confirmed empty photos.
Neither segmentation nor dataset collection requires all fragments of an
original page, a known page ID, or fragments from the same page. Reconstruction
inventory and page ownership remain the team's/orchestrator's responsibility.

The orchestrator audited for this handoff is commit
`6e2a4abb547a9c10d9ddff091fd6c565f62740e3`. Its current GUI, normalizer, acceptance
gate and final inventory verifier assume one photo equals one physical piece.
**Do not switch its Stage 1 pin to v2.5.0 until those assumptions are migrated.**
Its per-piece native boundary adapter remains applicable. The release's
`orchestrator_migration_required: true` metadata makes the required migration
explicit. This handoff does not claim a newly qualified four-stage Docker image.

Stage 2's input reader needs no change for the per-piece format. The reader in
its `main` commit `84c36bef1669ab421e6907051cf93cb10d087c53` is byte-identical to
the orchestrator-pinned `8b095867f1eea7216920196fd36c08a4f57b2820` reader.
After the existing native boundary adapter, it loaded all 28 pieces from seven
real photos, retaining 28 unique observation IDs and record fingerprints.
Multiple rows may share a source hash; Stage 2's per-piece paths/cache identities
use the observation ID or record fingerprint. The
[compatibility check](../provenance/STAGE2_V2_5_INPUT_COMPATIBILITY.json) records
the reader hash and validation scope. The migration below concerns orchestrator
source/piece accounting and review publication, while retaining that adapter.

## Invocation and exit codes

Install the v2.5.0 wheel with its `runtime` extra, or install the tagged source
with `pip install ".[runtime]"`. Unpack the model bundle, or run `git lfs pull`
in a source clone. The annotator's Windows ZIP already includes its model.

```text
python -m universal_segmenter.release_cli verify --models /absolute/models
python -m universal_segmenter.release_cli run --input /job/inputs/normalized --output /job/attempts/0001/stage1_native --models /absolute/models --capture-profile /job/capture_profile.json --review all
python -m universal_segmenter.release_cli inspect --input /job/attempts/0001/stage1_native
```

`paper-stage1` is the equivalent installed entry point. There is no GPU or Torch
requirement for inference. Stdout is one JSON result; progress goes to stderr.
Exit codes: ready `0`, failed `2`, review required `3`, interrupted `130`.
`--review all` leaves every candidate pending; `--review flagged` retains
automatic structural acceptance; `--review none` still flags structural failures.
This is a structural check, not a promise that every real piece was detected.

Use a new output for a changed job. `--resume` is only for the identical request,
including piece mode, input bytes, model, runtime and pipeline implementation.
Prediction caches are checksummed and include the all-components policy and
minimum native component area. Existing published artifacts are verified before
an identical resume returns its receipt. Do not resume a v2.0 job under v2.5.

## Per-piece wire contract

`segmentation_manifest.jsonl` retains schema
`paper-segmentation-manifest-1.0` and its native mask/crop fields, with additive
source-linkage fields. Its **cardinality has changed**: one source can have many
rows. Never require the number of manifest rows to equal the photo count.

| Field | Meaning |
| --- | --- |
| `capture_id` | Unique individual observation/piece ID; downstream Stage 2/3/4 use this ID. |
| `source_capture_id` | Identity of the original decoded input file/page, `capture_` plus `source_sha256`; shared by the photo's pieces. |
| `source_sha256` | Hash of that input file/page, not a unique piece key. |
| `candidate_id` | Stable model proposal identity within this source. |
| `artifact_id` | Unique filename key used when materializing this observation. |
| `source_path` | Absolute original native source path; multiple pieces share it. Keep it inside the job's immutable source storage. |
| `original_size_hw` | Original source size, unchanged for every piece from that source. |
| `bbox_xywh` | This piece's native crop origin and dimensions. |
| `mask_path` | Full native-size binary PNG containing only this piece. |
| `piece_path` | Tight native RGBA crop; alpha equals the cropped binary mask. |
| `piece_mask_path` | Native-size binary crop mask. |
| `overlay_path` | Individual review preview; preview resolution is not the geometry authority. |
| `accepted`, `review_status`, `reason_codes` | Individual observation review/acceptance state. |
| `empty_proposal` | True only for a pending source-review placeholder when the model proposed nothing. Never an accepted physical piece. |

For a source with one candidate, the historical `capture_<source hash>` ID is
retained. Multiple candidates receive source-prefixed proposal IDs. Treat IDs as
opaque strings; do not derive piece counts or filenames by parsing them.
Corrections preserve the original observation ID. A new user-created/split
observation must receive a new ID and retain its parent/source provenance.

**Critical review-publication fix:** the old orchestrator calls
`_materialize(target, row['source_sha256'], rgb, mask)`. With multiple pieces that
overwrites the other pieces from the same photo. Use
`row.get('artifact_id', row['source_sha256'])` instead, and keep the four returned
artifact paths in that observation's row.

The native masks of pieces from one source must not overlap. Their bounding
rectangles may overlap; this is normal for irregular shapes and is not a reason
to merge or discard pieces. Instance masks, not boxes, define the fragments.

## Source inventory and counts

`segmentation_run.json` retains schema `paper-segmentation-run-1.0` and adds:

```json
{
  "capture_mode": "multi",
  "complete": true,
  "total_discovered": 1,
  "source_count": 1,
  "piece_count": 2,
  "sources": [{
    "source_capture_id": "capture_<source hash>",
    "source_sha256": "<source hash>",
    "proposal_count": 2,
    "observation_ids": ["<piece A id>", "<piece B id>"],
    "empty_reviewed": false
  }]
}
```

`complete` means every discovered source was processed. It does not certify that
all fragments of a physical page exist. `piece_count` counts real proposals and
excludes empty placeholders. `inspect` reports source, piece, observation and
accepted counts separately.

A zero-proposal prediction emits an explicit unaccepted `empty_proposal` row,
not a silent success. The reconstruction GUI must let the operator confirm that
the source is empty or add missing pieces. For dataset collection, use the
annotator's **Confirm NO pieces** action; those are valid negative samples.

After review, rejected observations must be removed from the effective manifest
and each source's `observation_ids`, while retaining a rejection receipt/history.
A source with no retained observations needs an explicit empty confirmation
(`empty_reviewed: true`). Add/split operations must update the same inventory.
Only accepted, non-placeholder observations enter Stage 2. Check for missing
sources separately from the user's expected final physical piece count.

## Required orchestrator edits

1. `tools/stage_io.py:normalize`: normalize **captures**, not presumed pieces.
   Remove `len(captures) == expected_pieces`. Keep the input/pixel/PDF resource
   checks and duplicate-capture protection. Store the optional expected physical
   piece count separately and enforce it after detection/review.
2. `tools/stage_io.py:gate`: verify Stage 1 source coverage against normalized
   source hashes using `source_capture_id` and the run inventory. Independently
   check unique observation IDs, accepted masks, explicit empty confirmations,
   and the optional expected physical piece count. `verify_output()` checks the
   native crop/alpha geometry, source linkage, duplicates and mask overlap.
3. `app/api.py` and `web/`: group/display candidates by source photo, but edit,
   approve, reject and select each observation independently. Add missed-piece
   and split/merge operations with explicit new identities. Preserve separate
   masks and drafts for observations sharing a source. Offer source-level
   confirmation that every visible piece has been reviewed; do not demand a
   complete original page as part of Stage 1.
4. Review publication: fix `artifact_id` as above. Verify each correction against
   that observation's base-mask checksum. Require one connected unclipped mask;
   reject overlaps with other retained pieces from the same source. Update the
   effective source inventory after additions, splits, merges or rejections.
   Keep raw predictions immutable and reseal the effective publication/profile.
5. `tools/stage_io.py:verify4`: expected final `piece_id` values come from the
   accepted effective Stage 1 observations, not one `capture_` ID per uploaded
   file. Still enforce the user-declared physical inventory separately.
6. `app/worker.py`: retain headless review exit-code handling and immutable job
   profiles. Pass `--piece-mode multi` explicitly for clarity. Default to
   `--review all` for the initial multi-piece rollout. Keep existing single-piece
   job/review fixtures as regression controls; start new jobs for this version.
7. `composition/`: after the application changes and controls pass, pin Stage 1
   to the exact commit behind `v2.5.0`; regenerate compatibility/model/runtime
   snapshots, the composition lock, application identity and image receipt.
   Teach release discovery to reject an unsupported `capture_mode_default` /
   required migration before enabling stable-channel updates. Do not label a
   newly selected tuple/image qualified from metadata alone.

The existing `composition/adapter/prepare_stage1_for_stage2.py` remains useful:
it processes one manifest row at a time, preserves native scale, applies its
bounded connectivity cleanup, and adds a zero-mask crop margin. Preserve its raw
publication and review gates. Stage 1's tight crops intentionally do not add
that downstream margin themselves.

## Scale and physical identity

The network's 768-pixel letterbox is only an inference view. All exported masks,
crop offsets and pieces are restored to native pixels; no individual piece is
resized or perspective-corrected for downstream reconstruction. The orchestrator
must still enforce a common physical pixel scale: consistent scan DPI, or fixed
phone height, zoom, resolution and a parallel flat paper plane. Stage 1 does not
infer scale, establish page ownership, or deduplicate repeated physical pieces
across different photographs. The collection app's optional physical IDs help
group known repeat observations for dataset splitting.

## Controls before changing the production image

- One photo with 1, 2 and 7 separated pieces; several photos for one puzzle.
- Shared source paths with unique per-piece artifact paths and review drafts.
- Edit/approve/reject one piece without changing another; add a missed piece;
  split a bridge and merge/review fragments explicitly.
- Empty prediction, explicit empty confirmation, clipped masks and tiny false
  proposals; unknown and declared expected physical piece counts.
- Overlapping bounding boxes with disjoint masks; reject actual mask overlap.
- Resume/cancellation, stale correction, source mutation and artifact tampering.
- Boundary adapter, Stage 2 loading/identity, final Stage 4 ID inventory and a
  complete four-stage image/job control on a known puzzle.

Source verification helper:

```text
python scripts/check_orchestrator_handoff.py --orchestrator /path/to/orchestrator --input /path/to/separated-piece-photos --output /new/diagnostic-folder
```

It runs the actual CPU model, inspects the publication, applies the audited
orchestrator boundary adapter and verifies native crop/alpha invariants. It
does not claim to test the future GUI or solve/reconstruct a full page.
