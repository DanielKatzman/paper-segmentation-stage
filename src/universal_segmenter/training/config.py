"""Configuration shared by the editor-first training scripts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainingConfig:
    dataset_root: Path = Path("data/complete_dataset")
    experiments_root: Path = Path("experiments")
    run_name: str = "segformer_b1_detail_768_real_v3"
    model_type: str = "segformer_b1_detail"
    initial_checkpoint: Path | None = None
    resolution: int = 768
    epochs: int = 12
    physical_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    validation_batch_size: int = 1
    num_workers: int = 2
    pretrained_encoder: bool = True
    freeze_encoder_epochs: int = 2
    encoder_learning_rate: float = 3.0e-5
    decoder_learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    warmup_fraction: float = 0.05
    minimum_learning_rate_factor: float = 0.05
    gradient_clip_norm: float = 1.0
    threshold: float = 0.5
    random_seed: int = 104729
    epoch_samples: int | None = None
    checkpoint_every_optimizer_steps: int = 100
    progress_every_batches: int = 20
    resume: bool = True
    device: str = "cuda"
    category_weights: dict[str, float] = field(default_factory=lambda: {
        "real_capture": 0.25,
        "real_derived": 0.25,
        "procedural": 0.45,
        "background": 0.05,
    })

    @property
    def effective_batch_size(self) -> int:
        return self.physical_batch_size * self.gradient_accumulation_steps

    def validate(self) -> None:
        if self.model_type != "segformer_b1_detail":
            raise ValueError(f"unsupported model_type: {self.model_type}")
        if self.resolution < 128 or self.resolution % 32:
            raise ValueError("resolution must be at least 128 and divisible by 32")
        for name in ("epochs", "physical_batch_size",
                     "gradient_accumulation_steps", "validation_batch_size"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least one")
        if self.num_workers < 0 or self.freeze_encoder_epochs < 0:
            raise ValueError("num_workers and freeze_encoder_epochs cannot be negative")
        if self.freeze_encoder_epochs >= self.epochs:
            raise ValueError("freeze_encoder_epochs must be smaller than epochs")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between zero and one")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if not 0.0 < self.minimum_learning_rate_factor <= 1.0:
            raise ValueError("minimum_learning_rate_factor must be in (0, 1]")
        if self.encoder_learning_rate <= 0 or self.decoder_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.checkpoint_every_optimizer_steps < 1:
            raise ValueError("checkpoint interval must be at least one optimizer step")
        if self.progress_every_batches < 1:
            raise ValueError("progress interval must be at least one batch")
        if not self.run_name.strip() or Path(self.run_name).name != self.run_name:
            raise ValueError("run_name must be one nonempty folder name")
        if self.initial_checkpoint is not None and not str(self.initial_checkpoint).strip():
            raise ValueError("initial_checkpoint cannot be empty")
        if set(self.category_weights) != {
                "real_capture", "real_derived", "procedural", "background"}:
            raise ValueError("category_weights has unexpected names")
        if any(value < 0 for value in self.category_weights.values()):
            raise ValueError("category weights cannot be negative")
        if sum(self.category_weights.values()) <= 0:
            raise ValueError("at least one category weight must be positive")

    def serialized(self) -> dict[str, Any]:
        value = asdict(self)
        value["dataset_root"] = str(self.dataset_root)
        value["experiments_root"] = str(self.experiments_root)
        value["initial_checkpoint"] = (
            None if self.initial_checkpoint is None
            else str(self.initial_checkpoint)
        )
        value["effective_batch_size"] = self.effective_batch_size
        return value

    def immutable_hash(self) -> str:
        """Hash every setting that changes optimizer/model trajectory."""
        ignored = {
            "checkpoint_every_optimizer_steps", "progress_every_batches",
            "num_workers", "resume", "experiments_root",
        }
        value = {
            key: item for key, item in self.serialized().items()
            if key not in ignored
        }
        # Backward compatibility for v1 runs created before fine-tuning
        # initialization was added. A missing field and an explicitly disabled
        # field mean the same thing and must produce the same resume hash.
        if value.get("initial_checkpoint") is None:
            value.pop("initial_checkpoint", None)
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
