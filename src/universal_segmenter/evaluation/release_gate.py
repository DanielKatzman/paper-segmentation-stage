"""Calibration and untouched-holdout gates for the immutable CPU release."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from universal_segmenter.inference.cpu_runtime import (
    create_cpu_session,
    prepare_input,
    run_probability,
)
from universal_segmenter.inference.postprocess import apply_postprocess
from universal_segmenter.inference.preprocess import restore_probability
from universal_segmenter.metrics import binary_metrics, boundary_metrics
from universal_segmenter.training.freeze import read_sample_rows, verify_dataset_freeze


CALIBRATION_SCHEMA = "paper-segmentation-calibration-1.0"
HOLDOUT_SCHEMA = "paper-segmentation-holdout-1.0"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("L"), dtype=np.uint8) >= 128


def _probabilities(dataset_root: Path, rows: list[dict[str, str]],
                   onnx_path: Path, resolution: int,
                   cpu_threads: int) -> list[tuple[dict[str, str], np.ndarray, np.ndarray]]:
    session = create_cpu_session(onnx_path, cpu_threads)
    values = []
    for index, row in enumerate(rows, start=1):
        image = _load_rgb(dataset_root / row["image_path"])
        target = _load_mask(dataset_root / row["mask_path"])
        tensor, transform = prepare_input(image, resolution)
        output = run_probability(session, tensor)[0, 0]
        probability = restore_probability(output, transform)
        if probability.shape != target.shape:
            raise ValueError(f"shape mismatch for {row['sample_id']}")
        values.append((row, probability, target))
        print(f"[release] inference {index}/{len(rows)} {row['sample_id']}",
              flush=True)
    return values


def _row_metrics(sample_id: str, profile: str, prediction: np.ndarray,
                 target: np.ndarray) -> dict[str, Any]:
    metrics = binary_metrics(prediction, target)
    metrics.update(boundary_metrics(prediction, target, 2))
    # Names from boundary_metrics are unqualified here.
    boundary_f1 = float(metrics["boundary_f1"])
    catastrophic = bool(
        metrics["iou"] < 0.97
        or boundary_f1 < 0.90
        or max(metrics["false_positive_fraction"],
               metrics["false_negative_fraction"]) > 0.01)
    return {
        "sample_id": sample_id,
        "capture_profile": profile,
        "iou": float(metrics["iou"]),
        "boundary_f1_at_2px": boundary_f1,
        "false_positive_fraction": float(metrics["false_positive_fraction"]),
        "false_negative_fraction": float(metrics["false_negative_fraction"]),
        "catastrophic": catastrophic,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "count": len(rows),
        "median_iou": median(row["iou"] for row in rows),
        "mean_iou": mean(row["iou"] for row in rows),
        "mean_boundary_f1_at_2px": mean(
            row["boundary_f1_at_2px"] for row in rows),
        "catastrophic_count": sum(bool(row["catastrophic"]) for row in rows),
        "catastrophic_rate": mean(float(row["catastrophic"]) for row in rows),
    }


def _evaluate(values: Iterable[tuple[dict[str, str], np.ndarray, np.ndarray]],
              threshold: float, policy: str) -> dict[str, Any]:
    rows = [
        _row_metrics(
            row["sample_id"], row["capture_profile"],
            apply_postprocess(probability >= threshold, policy), target)
        for row, probability, target in values
    ]
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_profile[str(row["capture_profile"])].append(row)
    return {
        "overall": _summary(rows),
        "profiles": {
            name: _summary(group) for name, group in sorted(by_profile.items())},
        "per_image": rows,
    }


def calibrate(dataset_root: Path, onnx_path: Path, export_metadata: dict[str, Any],
              output_path: Path, cpu_threads: int = 7) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    freeze = verify_dataset_freeze(dataset_root)
    rows = [row for row in read_sample_rows(dataset_root)
            if row["split"] == "calibration" and row["sample_kind"] == "real_capture"]
    if not rows:
        raise ValueError("the frozen dataset has no real calibration captures")
    values = _probabilities(
        dataset_root, rows, Path(onnx_path),
        int(export_metadata["resolution"]), cpu_threads)
    candidates = []
    for policy in ("raw", "largest_non_border"):
        for threshold in np.arange(0.30, 0.701, 0.025):
            report = _evaluate(values, float(threshold), policy)
            overall = report["overall"]
            score = (
                0.50 * float(overall["median_iou"])
                + 0.50 * float(overall["mean_boundary_f1_at_2px"])
                - 0.50 * float(overall["catastrophic_rate"]))
            candidates.append({
                "threshold": float(round(float(threshold), 3)),
                "postprocess_policy": policy,
                "selection_score": score,
                **report,
            })
    best = max(candidates, key=lambda row: (
        row["selection_score"],
        -abs(row["threshold"] - 0.5),
        row["postprocess_policy"] == "raw",
    ))
    result = {
        "schema_version": CALIBRATION_SCHEMA,
        "dataset_fingerprint": freeze["fingerprint"],
        "split": "calibration",
        "sample_count": len(rows),
        "onnx_sha256": export_metadata["onnx_sha256"],
        "threshold": best["threshold"],
        "postprocess_policy": best["postprocess_policy"],
        "selection_score": best["selection_score"],
        "overall": best["overall"],
        "profiles": best["profiles"],
        "per_image": best["per_image"],
        "candidate_count": len(candidates),
    }
    _atomic_json(Path(output_path), result)
    return result


def _contact_sheet(values: list[tuple[dict[str, str], np.ndarray, np.ndarray]],
                   evaluation: dict[str, Any], threshold: float, policy: str,
                   output: Path) -> None:
    metric_by_id = {row["sample_id"]: row for row in evaluation["per_image"]}
    tiles = []
    for row, probability, target in values:
        prediction = apply_postprocess(probability >= threshold, policy)
        rgb = _load_rgb(Path(row["_dataset_root"]) / row["image_path"])
        shown = rgb.copy()
        shown[prediction] = np.clip(
            0.80 * shown[prediction] + 0.20 * np.asarray((20, 240, 40)),
            0, 255).astype(np.uint8)
        disagreement = prediction != target
        shown[disagreement] = (255, 40, 30)
        image = Image.fromarray(shown)
        image.thumbnail((420, 320), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (440, 370), "white")
        tile.paste(image, ((440 - image.width) // 2, 35))
        metric = metric_by_id[row["sample_id"]]
        ImageDraw.Draw(tile).text(
            (8, 8),
            f"{row['sample_id']} | IoU {metric['iou']:.4f} | "
            f"BF1 {metric['boundary_f1_at_2px']:.4f}",
            fill="black")
        tiles.append(tile)
    columns = min(3, len(tiles))
    rows_count = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 440, rows_count * 370), (230, 230, 230))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 440, (index // columns) * 370))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=93, subsampling=0)


def evaluate_holdout(dataset_root: Path, onnx_path: Path,
                     export_metadata: dict[str, Any], calibration: dict[str, Any],
                     output_path: Path, preview_path: Path,
                     cpu_threads: int = 7) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    freeze = verify_dataset_freeze(dataset_root)
    if calibration.get("dataset_fingerprint") != freeze["fingerprint"]:
        raise ValueError("calibration belongs to a different frozen dataset")
    if calibration.get("onnx_sha256") != export_metadata.get("onnx_sha256"):
        raise ValueError("calibration belongs to a different ONNX model")
    rows = [row for row in read_sample_rows(dataset_root)
            if row["split"] == "holdout" and row["sample_kind"] == "real_capture"]
    if not rows:
        raise ValueError("the frozen dataset has no real holdout captures")
    for row in rows:
        row["_dataset_root"] = str(dataset_root)
    values = _probabilities(
        dataset_root, rows, Path(onnx_path),
        int(export_metadata["resolution"]), cpu_threads)
    threshold = float(calibration["threshold"])
    policy = str(calibration["postprocess_policy"])
    evaluation = _evaluate(values, threshold, policy)
    overall = evaluation["overall"]
    profiles = evaluation["profiles"]
    gates = {
        "median_iou": {
            "value": overall["median_iou"], "required": 0.985,
            "passed": overall["median_iou"] >= 0.985},
        "mean_boundary_f1_at_2px": {
            "value": overall["mean_boundary_f1_at_2px"], "required": 0.95,
            "passed": overall["mean_boundary_f1_at_2px"] >= 0.95},
        "catastrophic_rate": {
            "value": overall["catastrophic_rate"], "required": 0.12,
            "passed": overall["catastrophic_rate"] <= 0.12},
        "worst_profile_median_iou": {
            "value": min(row["median_iou"] for row in profiles.values()),
            "required": 0.96,
            "passed": min(row["median_iou"] for row in profiles.values()) >= 0.96},
        "worst_profile_boundary_f1": {
            "value": min(row["mean_boundary_f1_at_2px"] for row in profiles.values()),
            "required": 0.90,
            "passed": min(row["mean_boundary_f1_at_2px"] for row in profiles.values()) >= 0.90},
    }
    result = {
        "schema_version": HOLDOUT_SCHEMA,
        "valid": all(row["passed"] for row in gates.values()),
        "dataset_fingerprint": freeze["fingerprint"],
        "split": "holdout",
        "sample_count": len(rows),
        "onnx_sha256": export_metadata["onnx_sha256"],
        "threshold": threshold,
        "postprocess_policy": policy,
        "overall": overall,
        "profiles": profiles,
        "gates": gates,
        "per_image": evaluation["per_image"],
    }
    _atomic_json(Path(output_path), result)
    _contact_sheet(values, evaluation, threshold, policy, Path(preview_path))
    return result

