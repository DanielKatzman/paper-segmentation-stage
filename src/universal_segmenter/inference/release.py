"""Load and run the immutable, calibrated CPU release."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from universal_segmenter.contracts import ReasonCode, SegmentationResult
from universal_segmenter.quality.structural import diagnose_mask, structural_reasons

from .candidates import FINAL_MODEL_RELEASE, MODEL_FILENAME, MODEL_KEY, RUN_NAME
from .cpu_runtime import create_cpu_session, prepare_input, run_probability
from .postprocess import POSTPROCESS_VERSION, apply_postprocess
from .preprocess import restore_probability


RELEASE_SCHEMA = "paper-segmentation-release-1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resource_root(project: Path | None = None) -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(project or Path(__file__).resolve().parents[3]).resolve()


def locate_release(project: Path | None = None) -> tuple[Path, Path]:
    root = resource_root(project)
    folders = [root / "models", root / "exports"]
    for folder in folders:
        model = folder / MODEL_FILENAME
        release = folder / "release.json"
        if model.is_file() and release.is_file():
            return model, release
    return folders[0] / MODEL_FILENAME, folders[0] / "release.json"


@dataclass(frozen=True, slots=True)
class ReleaseConfig:
    model_path: Path
    release_path: Path
    model_sha256: str
    model_version: str
    resolution: int
    threshold: float
    postprocess_policy: str
    dataset_fingerprint: str

    @property
    def cache_namespace(self) -> str:
        return (
            f"{self.model_version}_{self.model_sha256[:16]}_"
            f"t{round(self.threshold * 1000):03d}_{self.postprocess_policy}")


def load_release(project: Path | None = None,
                 require_holdout_pass: bool = True) -> ReleaseConfig:
    model_path, release_path = locate_release(project)
    if not model_path.is_file() or not release_path.is_file():
        raise FileNotFoundError(
            f"Production release is incomplete. Expected {model_path.name} and "
            f"release.json in {model_path.parent}.")
    metadata = json.loads(release_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != RELEASE_SCHEMA:
        raise ValueError(f"unsupported release metadata: {release_path}")
    if metadata.get("model_release") != FINAL_MODEL_RELEASE:
        raise ValueError("release.json is not the approved SegFormer v3 release")
    if metadata.get("model_type") != MODEL_KEY:
        raise ValueError("release.json refers to a different architecture")
    if metadata.get("checkpoint_run_name") != RUN_NAME:
        raise ValueError("release.json refers to a different training run")
    actual_hash = sha256_file(model_path)
    if metadata.get("onnx_sha256") != actual_hash:
        raise ValueError("ONNX checksum mismatch; rebuild the release")
    holdout = metadata.get("holdout_evaluation", {})
    if require_holdout_pass and not bool(holdout.get("valid", False)):
        raise ValueError(
            "This model has not passed the frozen holdout gate. Run "
            "evaluate_release.py and finalize_release.py first.")
    calibration = metadata.get("calibration", {})
    return ReleaseConfig(
        model_path=model_path.resolve(),
        release_path=release_path.resolve(),
        model_sha256=actual_hash,
        model_version=str(metadata["model_release"]),
        resolution=int(metadata.get("resolution", 768)),
        threshold=float(calibration.get("threshold", 0.5)),
        postprocess_policy=str(calibration.get("postprocess_policy", "raw")),
        dataset_fingerprint=str(metadata.get("dataset_fingerprint", "")),
    )


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return 0, 0, 1, 1
    left, top = int(xs.min()), int(ys.min())
    right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
    return left, top, right - left, bottom - top


def _quality(probability: np.ndarray, mask: np.ndarray,
             reason_count: int) -> float:
    if not mask.any():
        return 0.0
    selected = probability[mask]
    confidence = float(np.mean(np.maximum(selected, 1.0 - selected)))
    return float(np.clip(confidence - 0.12 * reason_count, 0.0, 1.0))


class Segmenter:
    """A mode-free CPU segmenter shared by the GUI and batch pipeline."""

    def __init__(self, project: Path | None = None, cpu_threads: int | None = None,
                 require_holdout_pass: bool = True,
                 release: ReleaseConfig | None = None) -> None:
        import os
        self.release = release or load_release(
            project, require_holdout_pass=require_holdout_pass)
        threads = max(1, min(8, int(
            cpu_threads if cpu_threads is not None else (os.cpu_count() or 2) - 1)))
        self.cpu_threads = threads
        self.session = create_cpu_session(self.release.model_path, threads)

    def probability(self, rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
        started = perf_counter()
        tensor, transform = prepare_input(rgb, self.release.resolution)
        prepared = perf_counter()
        network = run_probability(self.session, tensor)
        inferred = perf_counter()
        probability = restore_probability(network[0, 0], transform)
        restored = perf_counter()
        timings = {
            "preprocess": (prepared - started) * 1000.0,
            "inference": (inferred - prepared) * 1000.0,
            "restore": (restored - inferred) * 1000.0,
            "total": (restored - started) * 1000.0,
        }
        transform_value = {
            "original_hw": list(transform.original_hw),
            "canvas_hw": list(transform.canvas_hw),
            "left": transform.left,
            "top": transform.top,
            "resized_width": transform.resized_width,
            "resized_height": transform.resized_height,
        }
        return probability, transform_value, timings

    def segment(self, rgb: np.ndarray) -> SegmentationResult:
        probability, transform, timings = self.probability(rgb)
        raw = probability >= self.release.threshold
        mask = apply_postprocess(raw, self.release.postprocess_policy)
        diagnostics = diagnose_mask(mask)
        reasons = structural_reasons(diagnostics)
        accepted = not reasons
        result = SegmentationResult(
            mask=mask,
            probability_map=probability,
            contour_xy=[],
            bbox_xywh=_bbox(mask),
            quality_score=_quality(probability, mask, len(reasons)),
            accepted=accepted,
            reason_codes=reasons,
            diagnostics={
                "area_fraction": diagnostics.area_fraction,
                "component_count": diagnostics.component_count,
                "second_component_fraction": diagnostics.second_component_fraction,
                "hole_fraction": diagnostics.hole_fraction,
                "touches_border": diagnostics.touches_border,
                "raw_component_count": diagnose_mask(raw).component_count,
                "postprocess_version": POSTPROCESS_VERSION,
                "postprocess_policy": self.release.postprocess_policy,
                "threshold": self.release.threshold,
            },
            transform=transform,
            model_version=(
                f"{self.release.model_version}:{self.release.model_sha256[:16]}"),
            timings_ms=timings,
        )
        result.validate()
        return result
