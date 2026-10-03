"""Atomic best/last checkpoints with exact training cursor and RNG state."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CHECKPOINT_SCHEMA = "universal-paper-training-checkpoint-1.0"


@dataclass(slots=True)
class TrainingCursor:
    epoch: int = 0
    next_batch: int = 0
    global_optimizer_step: int = 0
    best_score: float = float("-inf")
    best_epoch: int = -1
    complete: bool = False

    def serialized(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "next_batch": self.next_batch,
            "global_optimizer_step": self.global_optimizer_step,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainingCursor":
        return cls(
            epoch=int(value["epoch"]),
            next_batch=int(value["next_batch"]),
            global_optimizer_step=int(value["global_optimizer_step"]),
            best_score=float(value["best_score"]),
            best_epoch=int(value["best_epoch"]),
            complete=bool(value.get("complete", False)),
        )


def capture_rng_state(torch_module) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch_module.get_rng_state(),
        "torch_cuda": None,
    }
    if torch_module.cuda.is_available():
        state["torch_cuda"] = torch_module.cuda.get_rng_state_all()
    return state


def _cpu_byte_tensor(torch_module, value):
    """Normalize RNG payloads after torch.load map-location remapping."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "to"):
        return value.to(device="cpu", dtype=torch_module.uint8)
    return torch_module.as_tensor(
        value, dtype=torch_module.uint8, device="cpu")


def restore_rng_state(torch_module, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_module.set_rng_state(
        _cpu_byte_tensor(torch_module, state["torch_cpu"]))
    if torch_module.cuda.is_available() and state.get("torch_cuda") is not None:
        torch_module.cuda.set_rng_state_all([
            _cpu_byte_tensor(torch_module, value)
            for value in state["torch_cuda"]
        ])


def atomic_torch_save(torch_module, value: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch_module.save(value, temporary)
    os.replace(temporary, path)


def build_checkpoint(
        *, kind: str, config: dict[str, Any], config_hash: str,
        dataset_fingerprint: str, cursor: TrainingCursor, model,
        optimizer, scheduler, history: list[dict[str, Any]],
        epoch_accumulator: dict[str, float | int], rng_state: dict[str, Any],
        ) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "checkpoint_kind": kind,
        "config": config,
        "config_hash": config_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "cursor": cursor.serialized(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": history,
        "epoch_accumulator": epoch_accumulator,
        "rng_state": rng_state,
    }


def validate_resume_checkpoint(value: dict[str, Any], config_hash: str,
                               dataset_fingerprint: str) -> None:
    if value.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported or corrupt training checkpoint")
    if value.get("checkpoint_kind") != "last":
        raise ValueError("resume must use last.pt, never best.pt")
    if value.get("config_hash") != config_hash:
        raise ValueError(
            "training settings changed since last.pt was saved. Restore the "
            "original settings or choose a new RUN_NAME.")
    if value.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError(
            "dataset changed since last.pt was saved. Restore the frozen "
            "dataset or choose a new RUN_NAME.")


def validate_initial_checkpoint(value: dict[str, Any], model_type: str,
                                resolution: int) -> None:
    """Validate weights used to initialize a new fine-tuning run."""
    if value.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported or corrupt initial training checkpoint")
    if value.get("checkpoint_kind") not in {"best", "last"}:
        raise ValueError("initial checkpoint must be a training best.pt or last.pt")
    source_config = value.get("config")
    if not isinstance(source_config, dict):
        raise ValueError("initial checkpoint has no training configuration")
    if source_config.get("model_type") != model_type:
        raise ValueError(
            "initial checkpoint model mismatch: expected "
            f"{model_type}, found {source_config.get('model_type')}")
    if int(source_config.get("resolution", -1)) != int(resolution):
        raise ValueError(
            "initial checkpoint resolution mismatch: expected "
            f"{resolution}, found {source_config.get('resolution')}")
    if not isinstance(value.get("model_state"), dict):
        raise ValueError("initial checkpoint has no model weights")
