"""Frozen identity and paths for the approved SegFormer v3 production model."""

from __future__ import annotations

from pathlib import Path

from .cpu_runtime import CpuModelSpec


FINAL_MODEL_RELEASE = "segformer-v3"
MODEL_KEY = "segformer_b1_detail"
RUN_NAME = "segformer_b1_detail_768_real_v3"
MODEL_FILENAME = "segformer_b1_detail_768_v3.onnx"


def final_model_spec(project: Path) -> CpuModelSpec:
    project = Path(project).resolve()
    return CpuModelSpec(
        name="SegFormer-B1 Detail v3",
        model_type=MODEL_KEY,
        resolution=768,
        checkpoint_run_name=RUN_NAME,
        checkpoint_path=(
            project / "experiments" / RUN_NAME / "checkpoints" / "best.pt"),
        onnx_path=project / "exports" / MODEL_FILENAME,
    )

