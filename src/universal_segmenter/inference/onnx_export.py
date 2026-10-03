"""Checkpoint-to-ONNX export with mandatory PyTorch/ONNX parity checks."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from universal_segmenter.models import create_model

from .candidates import FINAL_MODEL_RELEASE
from .cpu_runtime import CpuModelSpec, create_cpu_session, run_probability


EXPORT_SCHEMA = "paper-segmentation-onnx-export-1.0"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def export_is_current(spec: CpuModelSpec) -> bool:
    checkpoint = Path(spec.checkpoint_path)
    output = Path(spec.onnx_path)
    metadata_path = output.with_suffix(".json")
    if not checkpoint.is_file() or not output.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("schema_version") == EXPORT_SCHEMA
        and metadata.get("model_release") == FINAL_MODEL_RELEASE
        and metadata.get("model_type") == spec.model_type
        and metadata.get("checkpoint_run_name") == spec.checkpoint_run_name
        and int(metadata.get("resolution", -1)) == spec.resolution
        and metadata.get("checkpoint_path") == str(checkpoint.resolve())
        and int(metadata.get("checkpoint_size_bytes", -1)) == checkpoint.stat().st_size
        and int(metadata.get("checkpoint_mtime_ns", -1)) == checkpoint.stat().st_mtime_ns
    )


def _parity_input(resolution: int) -> np.ndarray:
    """A deterministic input containing flat, gradient, and sharp-edge regions."""
    coordinate = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    red = xx
    green = yy
    blue = np.where(xx + 0.35 * yy > 0.0, 0.75, -0.75).astype(np.float32)
    return np.ascontiguousarray(np.stack((red, green, blue), axis=0)[None])


def export_model(spec: CpuModelSpec, cpu_threads: int,
                 opset: int = 17) -> dict[str, Any]:
    checkpoint_path = Path(spec.checkpoint_path).resolve()
    output_path = Path(spec.onnx_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"best checkpoint is missing for {spec.name}: {checkpoint_path}")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("ONNX export requires the training PyTorch environment") from error
    torch.set_num_threads(max(1, int(cpu_threads)))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_kind") != "best":
        raise ValueError(f"export requires best.pt, received {checkpoint.get('checkpoint_kind')!r}")
    config = checkpoint.get("config", {})
    if config.get("model_type") != spec.model_type:
        raise ValueError(
            f"checkpoint model is {config.get('model_type')!r}, expected {spec.model_type!r}")
    if int(config.get("resolution", -1)) != spec.resolution:
        raise ValueError(
            f"checkpoint resolution is {config.get('resolution')!r}, "
            f"expected {spec.resolution}")
    if config.get("run_name") != spec.checkpoint_run_name:
        raise ValueError(
            f"checkpoint run is {config.get('run_name')!r}, expected "
            f"{spec.checkpoint_run_name!r}. Refusing to export an old model.")

    class ProbabilityWrapper(torch.nn.Module):
        def __init__(self, base_model) -> None:
            super().__init__()
            self.base_model = base_model

        def forward(self, value):
            return torch.sigmoid(self.base_model(value)["mask_logits"])

    model = create_model(spec.model_type, pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    wrapper = ProbabilityWrapper(model).eval()
    example = torch.from_numpy(_parity_input(spec.resolution))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    export_arguments: dict[str, Any] = {
        "input_names": ["image"],
        "output_names": ["mask_probability"],
        "opset_version": int(opset),
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_arguments["dynamo"] = False
    try:
        with torch.inference_mode():
            torch.onnx.export(wrapper, example, str(temporary), **export_arguments)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    with torch.inference_mode():
        torch_probability = wrapper(example).detach().cpu().numpy().astype(np.float32)
    session = create_cpu_session(output_path, cpu_threads)
    onnx_probability = run_probability(session, example.numpy())
    absolute = np.abs(torch_probability - onnx_probability)
    mask_agreement = float(np.mean(
        (torch_probability >= 0.5) == (onnx_probability >= 0.5)))
    parity = {
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "threshold_mask_agreement": mask_agreement,
    }
    # Probability parity is authoritative. Threshold agreement is recorded as a
    # useful diagnostic, but synthetic pixels extremely close to 0.5 can flip
    # without representing a meaningful export error.
    if (parity["max_absolute_error"] > 1.0e-3
            or parity["mean_absolute_error"] > 1.0e-5):
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ONNX parity failed for {spec.name}: {parity}. Export was removed.")

    cursor = checkpoint.get("cursor", {})
    metadata = {
        "schema_version": EXPORT_SCHEMA,
        "model_release": FINAL_MODEL_RELEASE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "name": spec.name,
        "model_type": spec.model_type,
        "resolution": spec.resolution,
        "input_name": "image",
        "input_shape": [1, 3, spec.resolution, spec.resolution],
        "output_name": "mask_probability",
        "output_shape": [1, 1, spec.resolution, spec.resolution],
        "threshold": float(config.get("threshold", 0.5)),
        "opset": int(opset),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_run_name": spec.checkpoint_run_name,
        "dataset_fingerprint": str(checkpoint.get("dataset_fingerprint", "")),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns,
        "best_epoch_zero_based": int(cursor.get("best_epoch", -1)),
        "best_epoch_one_based": int(cursor.get("best_epoch", -1)) + 1,
        "best_validation_score": float(cursor.get("best_score", float("nan"))),
        "onnx_path": str(output_path),
        "onnx_sha256": sha256_file(output_path),
        "onnx_size_bytes": output_path.stat().st_size,
        "parity": parity,
    }
    metadata_path = output_path.with_suffix(".json")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_metadata, metadata_path)
    return metadata


def ensure_export(spec: CpuModelSpec, cpu_threads: int,
                  opset: int = 17, force: bool = False) -> dict[str, Any]:
    if not force and export_is_current(spec):
        metadata_path = Path(spec.onnx_path).with_suffix(".json")
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return export_model(spec, cpu_threads=cpu_threads, opset=opset)
