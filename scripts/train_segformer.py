"""Reproducible, interruption-safe SegFormer-B1 v3 training recipe."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from universal_segmenter.training import TrainingConfig, run_training


DATASET_ROOT = Path("data/complete_dataset_v3")
RUN_NAME = "segformer_b1_detail_768_real_v3"
# This is the initialization used for the completed v3 run. Keep the v2 best
# checkpoint only if the run must be reproduced; it is not a production asset.
INITIAL_CHECKPOINT = Path(
    "experiments/segformer_b1_detail_768_real_v2/checkpoints/best.pt")


def build_config() -> TrainingConfig:
    return TrainingConfig(
        dataset_root=DATASET_ROOT,
        initial_checkpoint=INITIAL_CHECKPOINT,
        run_name=RUN_NAME,
        model_type="segformer_b1_detail",
        resolution=768,
        epochs=8,
        physical_batch_size=1,
        gradient_accumulation_steps=2,
        num_workers=2,
        freeze_encoder_epochs=1,
        encoder_learning_rate=1.0e-5,
        decoder_learning_rate=1.0e-4,
        category_weights={
            "real_capture": 0.45,
            "real_derived": 0.25,
            "procedural": 0.20,
            "background": 0.10,
        },
    )


if __name__ == "__main__":
    print(json.dumps(run_training(PROJECT, build_config()), indent=2))

