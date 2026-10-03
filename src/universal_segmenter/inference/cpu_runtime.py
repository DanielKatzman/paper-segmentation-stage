"""Shared CPU-only ONNX preprocessing, inference, and benchmark helpers."""

from __future__ import annotations

import hashlib
import math
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .preprocess import LetterboxTransform, letterbox, restore_probability


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
IMAGE_SUFFIXES = frozenset({
    ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
})
REQUIRED_SAMPLE_COLUMNS = {
    "sample_kind", "capture_profile", "image_path",
}


def _read_sample_rows(dataset_root: Path) -> list[dict[str, str]]:
    path = Path(dataset_root) / "samples.csv"
    if not path.is_file():
        raise FileNotFoundError(f"generated sample manifest is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_SAMPLE_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"samples.csv is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("samples.csv contains no samples")
    return rows


@dataclass(frozen=True, slots=True)
class CpuModelSpec:
    name: str
    model_type: str
    resolution: int
    checkpoint_run_name: str
    checkpoint_path: Path
    onnx_path: Path


def prepare_input(rgb: np.ndarray, resolution: int
                  ) -> tuple[np.ndarray, LetterboxTransform]:
    """Letterbox and normalize one RGB image into contiguous NCHW FP32."""
    boxed, transform = letterbox(rgb, int(resolution))
    value = boxed.astype(np.float32) / 255.0
    value = (value - IMAGENET_MEAN) / IMAGENET_STD
    tensor = np.ascontiguousarray(value.transpose(2, 0, 1)[None])
    return tensor, transform


def finish_prediction(probability: np.ndarray,
                      transform: LetterboxTransform,
                      threshold: float = 0.5
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Restore one model probability map and produce an 8-bit binary mask."""
    value = np.asarray(probability, dtype=np.float32)
    if value.ndim == 4 and value.shape[:2] == (1, 1):
        value = value[0, 0]
    elif value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or value.shape != transform.canvas_hw:
        raise ValueError(
            "model output must be one probability map matching the input canvas; "
            f"received {value.shape}, expected {transform.canvas_hw}")
    if not np.isfinite(value).all():
        raise ValueError("model output contains NaN or infinity")
    restored = restore_probability(value, transform)
    mask = np.where(restored >= float(threshold), 255, 0).astype(np.uint8)
    return restored, mask


def create_cpu_session(model_path: Path, cpu_threads: int):
    """Create the exact ONNX Runtime session used by the annotation tool."""
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "CPU inference requires onnxruntime. Install "
            "requirements-inference.txt (or requirements-export.txt).") from error
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = max(1, int(cpu_threads))
    options.enable_cpu_mem_arena = True
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"])


def run_probability(session, input_tensor: np.ndarray) -> np.ndarray:
    """Run one fixed-batch ONNX model and validate its probability output."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    if len(outputs) != 1:
        raise ValueError(f"expected one ONNX output, received {len(outputs)}")
    probability = np.asarray(outputs[0], dtype=np.float32)
    if probability.shape[:2] != (1, 1):
        raise ValueError(
            f"expected ONNX output [1, 1, H, W], received {probability.shape}")
    return probability


def duration_summary(milliseconds: Sequence[float]) -> dict[str, float | int]:
    if not milliseconds:
        raise ValueError("cannot summarize an empty duration list")
    value = np.asarray(milliseconds, dtype=np.float64)
    return {
        "count": int(value.size),
        "mean_ms": float(value.mean()),
        "median_ms": float(np.median(value)),
        "p90_ms": float(np.percentile(value, 90)),
        "p95_ms": float(np.percentile(value, 95)),
        "min_ms": float(value.min()),
        "max_ms": float(value.max()),
    }


def _is_source_image(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & {"masks", "overlays", "previews"}:
        return False
    stem = path.stem.lower()
    return not stem.endswith(("_mask", "_overlay", "_preview"))


def _stable_order(paths: Iterable[Path]) -> list[Path]:
    return sorted(
        {Path(path).resolve() for path in paths},
        key=lambda path: hashlib.sha256(str(path).encode("utf-8")).hexdigest())


def _round_robin(groups: dict[str, list[Path]], limit: int) -> list[Path]:
    ordered = {name: _stable_order(paths) for name, paths in groups.items()}
    names = sorted(name for name, paths in ordered.items() if paths)
    result: list[Path] = []
    position = 0
    while names and len(result) < limit:
        remaining: list[str] = []
        for name in names:
            paths = ordered[name]
            if position < len(paths):
                result.append(paths[position])
                if len(result) == limit:
                    break
            if position + 1 < len(paths):
                remaining.append(name)
        names = remaining
        position += 1
    return result


def discover_benchmark_images(dataset_root: Path,
                              input_folder: Path | None,
                              limit: int) -> list[Path]:
    """Choose stable, profile-balanced real captures for fair model timing."""
    if int(limit) < 1:
        raise ValueError("benchmark image limit must be positive")
    if input_folder is not None:
        folder = Path(input_folder).resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"benchmark input folder does not exist: {folder}")
        grouped: dict[str, list[Path]] = defaultdict(list)
        for path in folder.rglob("*"):
            if not _is_source_image(path):
                continue
            relative = path.relative_to(folder)
            profile = relative.parts[0] if len(relative.parts) > 1 else "root"
            grouped[profile].append(path)
        images = _round_robin(dict(grouped), limit)
        if not images:
            raise ValueError(f"no supported source images found in {folder}")
        return images

    root = Path(dataset_root).resolve()
    rows = _read_sample_rows(root)
    by_profile: dict[str, list[Path]] = defaultdict(list)
    for row in rows:
        if row["sample_kind"] != "real_capture":
            continue
        path = (root / row["image_path"]).resolve()
        if _is_source_image(path):
            by_profile[row["capture_profile"]].append(path)
    if not by_profile:
        raise ValueError(
            "samples.csv contains no usable real_capture images. Set "
            "BENCHMARK_INPUT_FOLDER at the top of the benchmark script.")
    images = _round_robin(dict(by_profile), limit)
    if not images:
        raise ValueError("no usable real capture images were found")
    return images


def speed_recommendation(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Rank models only by measured median end-to-end latency."""
    if len(results) < 2:
        raise ValueError("at least two model results are required")
    ranked = sorted(
        results,
        key=lambda result: float(result["timings"]["end_to_end"]["median_ms"]))
    fastest, second = ranked[0], ranked[1]
    fastest_ms = float(fastest["timings"]["end_to_end"]["median_ms"])
    second_ms = float(second["timings"]["end_to_end"]["median_ms"])
    ratio = math.inf if fastest_ms <= 0 else second_ms / fastest_ms
    return {
        "default_model": fastest["name"],
        "optional_model": second["name"],
        "secondary_to_default_latency_ratio": ratio,
        "basis": "lowest median warm-cache end-to-end CPU latency",
        "quality_note": (
            "This is a speed recommendation only. Final tool defaults must also "
            "use the completed validation and real holdout quality results."),
    }
