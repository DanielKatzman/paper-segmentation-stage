# Current status and Tesla P40 training plan

Prepared 6 November 2026. Scope: explain the evidence, plan the next model-training stage, and identify checks still needed on the university machine. No university-machine access or new model training has taken place.

Hardware update from the latest screenshots: **the VM exposes GRID P40-8Q with 8,192 MiB VRAM, not the full physical P40's 24 GB.** The query showed 6,433 MiB free. Driver is 535.161.08; the machine has Ubuntu 24.04.4 LTS, eight logical CPUs, 31 GiB system RAM, and about 170 GiB free disk. The plan below now uses this 8 GiB GPU allocation. CUDA execution through PyTorch still needs its own test.

## 1. Answer: the current model is not yet proven consistently accurate on multiple pieces

We have changed the software around the network, not its learned weights. The annotator now exposes multiple candidate components instead of discarding everything except one target. This makes the existing SegFormer v3 useful for initial suggestions. It does not establish that every candidate is correct or every piece is detected.

The successful packaged-app control used four simple synthetic rectangular pieces. Its purpose was to prove CPU inference and the collection/export/import/publication path. It reported four proposals but did not perform a real multi-piece accuracy benchmark.

The generated example contains eight scenes and thirty ground-truth instances. The generator already knows those masks; generating thirty correct labels is not the same as the model predicting thirty correct masks. Likewise, the 32-piece rendering and 256-instance storage tests establish software capabilities, not recognition accuracy.

The earlier four-image diagnostic, built from reviewed production crops, measured the following before the old one-target filter:

| Actual pieces | Binary union IoU | Ground-truth pieces matched one-to-one at IoU >= 0.5 |
| --- | --- | --- |
| 1 | 0.954 | 1/1 |
| 2 | 0.942 | 2/2 |
| 4 | 0.644 | 4/4 |
| 6 | 0.568 | 4/6 |

The four-piece case illustrates why count alone is insufficient: all four pieces had a loose match, but masks were substantially incomplete and fragmented. The raw maps also included extra components. These are only four synthetic layouts, not a statistically meaningful real-photo benchmark or a supported maximum count.

Conclusion: keep v3 as an annotation assistant and baseline; plan to fine-tune it with exhaustive multi-piece labels and evaluate small-piece/gap failures explicitly. All current proposals still need human review.

Evidence: the earlier review's `docs/reviews/stage1_multi_piece_2026-11-06/multi_piece_probe.json` in the parent research workspace, plus this checkout's `provenance/FOUNDATION_VALIDATION.json` and `src/universal_segmenter/scenes/selfcheck.py`.

## 2. Where the project stands

| Area | Current state |
| --- | --- |
| Original single-piece dataset | Preserved: 3,879 samples; frozen fingerprint verified unchanged. |
| Production network | Original SegFormer-B1 Detail v3 ONNX and weights; no multi-piece fine-tuning yet. |
| Multi-piece scene and review architecture | Implemented, including native labels, independent identities, completeness/empty states, and revision history. |
| Generator | Implemented: PDF fragments, real cutouts, retexturing, mixed scenes, background-only scenes, controlled placement, source eligibility and recipes. |
| Teammate app | Windows package built; extracted CPU/data control and GUI startup passed. Native editor/review/undo controls passed. Desktop screenshot inspection could not complete. |
| Collection exchange | Implemented: checked portable ZIPs, duplicate/conflict detection, explicit revision selection and immutable imported packages. |
| Dataset assembly | Implemented: central grouping, protected splits, content freeze, legacy compatibility and loader support. |
| Tests | 39 passed at the end of the foundation implementation. |
| New real multi-piece collection | No imported real collection in this checkout's `data/multi_piece/collections` as checked today. |
| New training dataset | No frozen real multi-piece dataset in `data/multi_piece/datasets` yet; example datasets are synthetic controls. |
| P40 environment | GRID P40-8Q, 8 GiB allocated VRAM, driver 535.161.08 confirmed by screenshots; PyTorch CUDA execution and throughput remain unverified. |
| New trained model / full orchestrator release | Still to do after data collection, profiling, training and qualification. |

The foundation is ready for data collection. The next model is not yet trained or accuracy-qualified. The local working branch is `codex/multi-piece-foundation`; the original production checkout and datasets remain separate.

## 3. What the screenshots establish

The first screenshot identifies `GP102GL [Tesla P40]` through `lspci`. This establishes the PCI GPU identity, not that the NVIDIA driver or CUDA training currently works.

The second screenshot is `free -h`: approximately 31 GiB total system RAM, with about 28 GiB available at capture time. This is consistent with a machine provisioned with approximately 32 GB of CPU/system RAM. It is not a GPU-memory measurement.

NVIDIA specifies the full Tesla P40 with **24 GB GDDR5 GPU memory**, and lists its compute capability as **6.1**. The later `nvidia-smi` output confirms this account receives an **8 GiB GRID P40-8Q virtual GPU allocation**, with 6,433 MiB free at query time. Use that allocation, not the physical card's capacity, for training-memory planning. [NVIDIA P40 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/solutions/resources/documents1/nvidia-p40-datasheet.pdf), [NVIDIA legacy GPU table](https://developer.nvidia.com/cuda/gpus/legacy).

Planning judgment: this 8 GiB allocation is worth testing for fine-tuning the existing model, but provides no nominal VRAM increase over the earlier 8 GB GTX 1070. Desktop/virtualization use also consumes some memory, and compute throughput must be measured. Start with physical batch 1 at 768 in FP32 and gradient accumulation. If one full training step exceeds the available memory, investigate smaller training views or validated memory-saving changes before proceeding. Do not assume larger batches or 1024-pixel inputs fit.

## 4. Milestone A: verify and reproduce the training environment

Run on the university machine:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
python3 --version
nproc
df -h .
```

Also establish whether the GPU is shared, the allowed job duration, persistent storage/quota, and how checkpoints should be retrieved. Keep a new isolated environment and project directory; do not alter a shared system driver or overwrite the working Windows environment.

The team confirmed Python can be installed freely on the university machine. Use an isolated Python 3.12 environment; the currently installed Python version is therefore not a blocker. The driver and allocation report has now been supplied. The local environment previously used for the existing project is:

```text
Python        3.12.10
PyTorch       2.7.1+cu118
CUDA runtime  11.8
Transformers  4.52.1
NumPy         2.5.2
```

Start by reproducing this training stack on Linux, subject to driver compatibility. PyTorch's official archive provides a 2.7.1 CUDA 11.8 wheel command. This is a compatibility/reproducibility choice, not a claim that this is the latest PyTorch release. [Official previous-version installation instructions](https://pytorch.org/get-started/previous-versions/).

Driver 535.161.08 is expected to run CUDA 11.8 applications through backward compatibility. The `CUDA Version: 12.2` line in `nvidia-smi` describes driver capability, not a requirement to install a CUDA 12.2 PyTorch wheel. Leave the university's vGPU driver unchanged and test the CUDA 11.8 wheel. [NVIDIA CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). Exact setup and a repository-independent GPU test are in [University GPU setup](UNIVERSITY_GPU_SETUP.md).

Do not simply install the newest default CUDA wheel: Pascal support depends on the binary's architecture coverage. PyTorch explicitly removed Pascal from its CUDA 12.8 builds starting with release 2.8. [PyTorch packaging announcement](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128).

After environment creation, check `torch.cuda.is_available()`, the reported device/memory/compute capability, and `torch.cuda.get_arch_list()`. Then run an actual CUDA forward/backward/optimizer step with the project's model. Architecture-list or GPU-detection checks alone do not prove all required kernels work. Save exact package versions and the hardware report.

Completion gate: a repeated real training-step smoke test succeeds on CUDA with finite loss, and a saved checkpoint can be loaded. Establish throughput only after that check.

## 5. Milestone B: collect a pilot and establish the unchanged-model baseline

Give teammates the existing collection tool now. Collect exhaustive annotations for separated, fully visible pieces and confirmed-empty images. Include both same-DPI scans and fixed-height/parallel phone captures, with varied text, fragment shapes, sizes, lighting, backgrounds and visible gap widths.

Use count bins 0, 1, 2–4, 5–8, 9–16 and 17–32 initially; larger counts are stress cases until their piece-size/gap envelope is understood. A pilot target is roughly 100–200 completely annotated real scenes across multiple physical pages and sessions. Expand based on failures; photo count alone is not sufficient diversity. Reserve independent new pages/sessions for the final holdout before tuning.

Measure v3 on an exploratory real benchmark first: missed pieces, false proposals, split/merged masks, per-piece IoU and boundary quality, plus review time. Keep the exploratory set distinct from the final untouched holdout. This determines how much improvement is needed and which backgrounds/count/size ranges require more labels.

Use generated scenes for controlled training variety and diagnostic coverage. Start with a manageable pilot generation, inspect previews and derived-resolution warnings, then scale generation. More synthetic images do not replace independent real evaluation.

Completion gate: a real baseline report with a gallery of missed, merged, fragmented and poor-boundary cases; a split-safe collection inventory; and no incomplete scenes masquerading as negative labels.

## 6. Milestone C: assemble and freeze the first actual training dataset

Create a new dataset version alongside v3. Combine eligible legacy training examples, new real scenes, generated multi-piece scenes and a controlled negative share. Audit legacy images containing unlabeled paper distractors before reuse. Never move historical validation/calibration/holdout captures into training.

Keep complete page/physical-piece families together and retain session/background provenance. Only training-eligible cutouts, texture PDFs and backgrounds may produce generated training scenes. The builder must show real captures in all four splits before the training qualification gate is satisfied.

Initial sampling should preserve both single-piece performance and multi-piece coverage. Configure the real/procedural/derived/negative mixture and optional count-bin weights from actual available data. Keep the negative draw proportion around 10% as a starting experiment, not a fixed requirement; avoid balancing mistakes that let easy backgrounds or one dense scene dominate training.

Keep native/master images and labels. Use 768-pixel views for the first comparable model. For scenes whose small pieces/gaps disappear at that resolution, plan appropriate tile views or collect higher-detail captures; do not expect upsampling to recover missing information.

Completion gate: content-frozen dataset, audited provenance/splits, coverage counts by acquisition/count/minimum size/gap, readable loader samples, and a fixed real validation/calibration/holdout protocol.

## 7. Milestone D: finish the new-run controls and profile on the P40

The architecture/data foundation is present, but the historical `scripts/train_segformer.py` still names v3 and its old initialization checkpoint. Before a long new run:

1. Add a dedicated multi-piece training entry/configuration with the new frozen dataset, a new run name, and v3 `best.pt` as initialization. Do not resume or overwrite the completed v3 run.
2. Add a repeatable P40 profile reporting full training-step time, images/sec, actual GPU/RAM usage and peak allocated/reserved VRAM. Profile after encoder unfreezing and optimizer-state allocation, not just an encoder-frozen forward pass.
3. Promote instance recall/merge/miss criteria into model-selection/release gates. Instance metrics are logged today, but the inherited selection score still primarily uses union/boundary metrics.
4. Prepare multi-piece calibration/evaluation commands. The historical release calibration/export/finalization scripts remain pinned to v3 and single-target assumptions; they are not a ready-made new-model promotion path.
5. If a higher input resolution is selected later, explicitly adapt and verify weight initialization across resolutions. The current checkpoint guard requires matching resolution; do not bypass it without validating the model's transfer behavior.

Start profiling in FP32, matching the current working training implementation. Test physical batch 1 at 768 first, with roughly 15–20% headroom relative to the memory actually available. Use gradient accumulation for the desired effective batch. Only try batch 2 if measurements show it can fit; batch 4 and 1024 inputs are optional later experiments, not initial expectations under this 8 GiB virtual allocation. If batch 1 fails, first establish a feasible training-view size and checkpoint initialization path.

Start with two data-loader workers and on-demand image loading under the reported 32 GB system RAM; then measure whether CPU loading limits GPU use. Keep data on persistent local storage where possible and avoid holding the full native-resolution dataset in RAM.

Only after establishing the 768 baseline, profile 1024 inputs and/or 768 tile views. Compare methods under measured step/time budgets and the same real evaluation split. A mixed-precision experiment is optional; do not assume it is faster or necessary on this card.

Completion gate: documented compatible environment, measured batch/worker/resolution policy, validated checkpoint initialization, correct multi-piece selection metrics and a short resumable pilot run. No training-time estimate is credible before throughput and dataset draw counts are measured.

## 8. Milestone E: fine-tune and evaluate controlled candidates

Keep SegFormer-B1 Detail initially. Use the approved v3 checkpoint to start a new run. A starting fine-tuning recipe can retain the previous low learning rates (encoder about 1e-5, decoder about 1e-4), a brief encoder freeze and the existing compound loss, subject to the pilot's validation behavior.

Compare:

1. Unchanged v3 with the new candidate-extraction path.
2. Fine-tuned model at 768 whole-image input.
3. A higher-detail candidate: 1024 input or a validated tile-trained path, chosen after profiling and failure analysis.

Use validation/early stopping and retain best/last checkpoints with dataset and configuration identities. Review whether pieces disappear, join, split, or have damaged contours; a high union IoU cannot compensate for a missing physical fragment.

For dense-image inference, overlapping tiling and crop refinement need their own implementation and tests. Restore all probabilities to source coordinates, handle internal tile borders as partial views rather than clipped captures, blend overlapping evidence, and deduplicate candidates. The foundation's tile-view loader is useful infrastructure; it is not a completed tiled production inference engine.

Do not escalate architecture size before testing whether data diversity and preserved image detail solve the actual failures. Increase data/targeted sampling first when the failure gallery shows clear coverage gaps.

Completion gate: controlled validation comparison showing the multi-piece improvement and single-piece regression, together with runtime/memory/review-effort costs.

## 9. Milestone F: lock, test once on fresh holdout, and publish

Select model/inference policy on validation and thresholds on calibration, then evaluate the locked system on the fresh real holdout. Report sample counts and per-profile/count/size/gap results, not one aggregate percentage.

Report one-to-one instance precision/recall at IoU 0.5 and 0.75, exact-count rate, miss/merge/split/duplicate rates, per-instance boundary quality, negative false detections, and single-piece regression. Specify boundary-distance units and source/model resolution so comparisons are meaningful.

Provisional team targets, not current results: at least 98% recall at IoU 0.5, at least 95% at IoU 0.75, and at least 95% proposal precision inside an explicitly measured capture envelope. Require strong boundary quality and no material single-piece regression. Manual approval remains the publication default even when the automated model improves.

Export the selected checkpoint to ONNX and check parity, new model/release identity, CPU performance and the portable annotator. Then migrate and qualify the orchestrator's source/candidate/physical inventory and reviewed-publication route. Its complete-page/scale assumptions and current piece limits still apply until that integration work passes.

Completion gate: new qualified model and composition, documented limits, protected regression evidence, and a working rollback to the existing release.

## 10. Immediate next actions

1. Install the isolated training environment and run a CUDA forward/backward/optimizer smoke test; GPU, driver, CPU, RAM and disk information have now been supplied.
2. Begin real annotation collection with the delivered Windows tool while the Linux environment is prepared.
3. Run the exploratory unchanged-v3 real baseline and a small generator/loader pilot.
4. Freeze the first training dataset, finish the dedicated new-run/profile/evaluation controls, and run the P40 training-step benchmark.
5. Choose the actual training budget and configuration from those measurements; then start fine-tuning.

The machine details improve the training plan, but they do not replace the missing real multi-piece evidence. The next accuracy improvement comes from the collected labels, appropriate image detail, and a measured training/evaluation cycle.
