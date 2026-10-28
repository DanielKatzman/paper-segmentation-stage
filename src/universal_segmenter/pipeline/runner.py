"""Stable disk-to-disk Stage-1 pipeline for the reconstruction stages."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from universal_segmenter.annotation.editor import MaskEditor, create_overlay
from universal_segmenter.data.manifest import (
    read_jsonl,
    sha256_file,
    sync_capture_manifest,
    write_jsonl,
)
from universal_segmenter.inference.preprocess import load_rgb
from universal_segmenter.inference.release import Segmenter
from universal_segmenter.quality.structural import diagnose_mask, structural_reasons


PIPELINE_SCHEMA = "paper-segmentation-manifest-1.0"
ReviewMode = Literal["none", "flagged", "all"]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    mode = "L" if array.ndim == 2 else ("RGBA" if array.shape[2] == 4 else "RGB")
    Image.fromarray(array, mode=mode).save(temporary, format="PNG", compress_level=1)
    os.replace(temporary, path)


def _save_overlay(path: Path, rgb: np.ndarray, mask: np.ndarray) -> None:
    height, width = rgb.shape[:2]
    scale = min(1.0, 1600 / max(height, width))
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        shown_rgb = np.asarray(Image.fromarray(rgb).resize(
            size, Image.Resampling.LANCZOS), dtype=np.uint8)
        shown_mask = np.asarray(Image.fromarray(
            mask.astype(np.uint8) * 255).resize(
                size, Image.Resampling.NEAREST), dtype=np.uint8) >= 128
    else:
        shown_rgb, shown_mask = rgb, mask
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(create_overlay(shown_rgb, shown_mask)).save(
        temporary, format="JPEG", quality=94, subsampling=0)
    os.replace(temporary, path)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return 0, 0, 1, 1
    left, top = int(xs.min()), int(ys.min())
    right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
    return left, top, right - left, bottom - top


def _paths(output: Path, source_hash: str) -> dict[str, Path]:
    return {
        "mask": output / "masks" / f"{source_hash}.png",
        "piece": output / "pieces" / f"{source_hash}.png",
        "piece_mask": output / "piece_masks" / f"{source_hash}.png",
        "overlay": output / "overlays" / f"{source_hash}.jpg",
    }


def _cache_paths(output: Path, namespace: str,
                 source_hash: str) -> tuple[Path, Path]:
    root = output / "cache" / namespace
    return root / f"{source_hash}.png", root / f"{source_hash}.json"


def _materialize(output: Path, source_hash: str, rgb: np.ndarray,
                 mask: np.ndarray) -> dict[str, str]:
    paths = _paths(output, source_hash)
    _save_png(paths["mask"], mask.astype(np.uint8) * 255)
    left, top, width, height = _bbox(mask)
    crop = rgb[top:top + height, left:left + width]
    crop_mask = mask[top:top + height, left:left + width]
    rgba = np.dstack((crop, crop_mask.astype(np.uint8) * 255))
    _save_png(paths["piece"], rgba)
    _save_png(paths["piece_mask"], crop_mask.astype(np.uint8) * 255)
    _save_overlay(paths["overlay"], rgb, mask)
    return {key: path.relative_to(output).as_posix() for key, path in paths.items()}


def _cache_prediction(segmenter: Segmenter, output: Path, source_hash: str,
                      rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any], bool]:
    mask_path, metadata_path = _cache_paths(
        output, segmenter.release.cache_namespace, source_hash)
    if mask_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            checksum_matches = metadata.get("cache_mask_sha256") == sha256_file(mask_path)
            with Image.open(mask_path) as source:
                mask = np.asarray(source.convert("L"), dtype=np.uint8) >= 128
            if checksum_matches and mask.shape == rgb.shape[:2]:
                return mask, metadata, True
        except (OSError, ValueError):
            pass  # An interrupted or corrupted cache is recomputed from its source.
    result = segmenter.segment(rgb)
    _save_png(mask_path, result.mask.astype(np.uint8) * 255)
    metadata = result.metadata()
    metadata["cache_mask_sha256"] = sha256_file(mask_path)
    _atomic_json(metadata_path, metadata)
    return result.mask, metadata, False


def _existing_is_current(row: dict[str, Any], output: Path, source_hash: str,
                         model_version: str) -> bool:
    if row.get("source_sha256") != source_hash:
        return False
    if row.get("model_version") != model_version:
        return False
    return all((output / str(row.get(key, ""))).is_file() for key in (
        "mask_path", "piece_path", "piece_mask_path", "overlay_path"))


def run_pipeline(input_folder: Path, output_folder: Path,
                 review: ReviewMode = "flagged", cpu_threads: int | None = None,
                 pdf_dpi: int = 300, reprocess: bool = False,
                 project: Path | None = None, headless: bool = False,
                 release_directory: Path | None = None) -> dict[str, Any]:
    """Segment a folder recursively and emit the immutable Stage-1 contract."""
    if review not in {"none", "flagged", "all"}:
        raise ValueError("review must be none, flagged, or all")
    input_folder = Path(input_folder).resolve()
    output = Path(output_folder).resolve()
    if not input_folder.is_dir():
        raise FileNotFoundError(input_folder)
    if (input_folder == output or input_folder in output.parents
            or output in input_folder.parents):
        raise ValueError("output_folder cannot overlap input_folder")
    output.mkdir(parents=True, exist_ok=True)
    capture_manifest = output / ".state" / "captures.jsonl"
    scan = sync_capture_manifest(input_folder, capture_manifest, output)
    captures = read_jsonl(capture_manifest)
    segmenter = Segmenter(project=project, cpu_threads=cpu_threads,
                          release_directory=release_directory)
    model_version = (
        f"{segmenter.release.model_version}:{segmenter.release.model_sha256[:16]}")
    manifest_path = output / "segmentation_manifest.jsonl"
    existing = {
        str(row["source_sha256"]): row for row in read_jsonl(manifest_path)
        if row.get("source_sha256")
    }
    rows: list[dict[str, Any]] = []
    generated = reused = manual = flagged = 0
    stopped = False

    for index, capture in enumerate(captures, start=1):
        source_hash = str(capture["source_sha256"])
        previous = existing.get(source_hash)
        if (not reprocess and previous is not None
                and _existing_is_current(
                    previous, output, source_hash, model_version)):
            rows.append(previous)
            reused += 1
            print(f"[segment] {index}/{len(captures)} reused {source_hash[:12]}",
                  flush=True)
            continue
        source_path = Path(str(capture["image_path"]))
        if not source_path.is_absolute():
            source_path = output / source_path
        rgb = load_rgb(source_path, pdf_dpi=pdf_dpi)
        auto_mask, auto_metadata, cache_reused = _cache_prediction(
            segmenter, output, source_hash, rgb)
        reasons = list(auto_metadata.get("reason_codes", []))
        should_review = review == "all" or (review == "flagged" and bool(reasons))
        mask = auto_mask.copy()
        decision_source = "automatic"
        changed_fraction = 0.0
        if should_review and not headless:
            editor = MaskEditor(
                rgb, mask, auto_mask,
                f"Paper segmentation {index}/{len(captures)} - {source_hash[:12]}",
                proposal_key="segformer_b1_detail",
                proposal_name="SegFormer-B1 Detail v3",
                proposal_choices=[(
                    "segformer_b1_detail", "SegFormer-B1 Detail v3")],
                dataset_split="production",
                dataset_split_choices=[("production", "Production")],
            )
            action, corrected = editor.run()
            if action == "quit":
                stopped = True
                break
            if not corrected.any() or corrected.all():
                raise RuntimeError("refusing an empty or full corrected mask")
            mask = corrected
            changed_fraction = float(np.mean(mask != auto_mask))
            decision_source = "manual_review"
            manual += 1

        diagnostics = diagnose_mask(mask)
        final_reasons = [] if decision_source == "manual_review" else [
            str(reason) for reason in structural_reasons(diagnostics)]
        if headless and should_review:
            final_reasons = sorted(set(final_reasons + reasons + ["MANUAL_REVIEW_REQUIRED"]))
        accepted = not final_reasons
        if not accepted:
            flagged += 1
        artifact_paths = _materialize(output, source_hash, rgb, mask)
        row = {
            "schema_version": PIPELINE_SCHEMA,
            "capture_id": str(capture["capture_id"]),
            "source_path": str(source_path.resolve()),
            "source_sha256": source_hash,
            "capture_type_hint": str(capture.get("capture_type", "unknown")),
            "original_size_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
            "bbox_xywh": list(_bbox(mask)),
            "mask_path": artifact_paths["mask"],
            "piece_path": artifact_paths["piece"],
            "piece_mask_path": artifact_paths["piece_mask"],
            "overlay_path": artifact_paths["overlay"],
            "model_release": segmenter.release.model_version,
            "model_version": model_version,
            "model_sha256": segmenter.release.model_sha256,
            "dataset_fingerprint": segmenter.release.dataset_fingerprint,
            "threshold": segmenter.release.threshold,
            "postprocess_policy": segmenter.release.postprocess_policy,
            "decision_source": decision_source,
            "review_status": (
                "manually_reviewed" if decision_source == "manual_review"
                else ("auto_accepted" if accepted else "needs_review")),
            "accepted": accepted,
            "reason_codes": final_reasons,
            "manual_changed_fraction": changed_fraction,
            "diagnostics": {
                "area_fraction": diagnostics.area_fraction,
                "component_count": diagnostics.component_count,
                "second_component_fraction": diagnostics.second_component_fraction,
                "hole_fraction": diagnostics.hole_fraction,
                "touches_border": diagnostics.touches_border,
            },
            "cache_reused": cache_reused,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)
        rows.sort(key=lambda item: str(item["source_sha256"]))
        write_jsonl(manifest_path, rows)
        generated += 1
        print(
            f"[segment] {index}/{len(captures)} "
            f"{row['review_status']} {source_hash[:12]}", flush=True)

    rows.sort(key=lambda item: str(item["source_sha256"]))
    write_jsonl(manifest_path, rows)
    report = {
        "schema_version": "paper-segmentation-run-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": not stopped and len(rows) == len(captures),
        "stopped_safely": stopped,
        "input_folder": str(input_folder),
        "output_folder": str(output),
        "review_mode": review,
        "model_version": model_version,
        "folder_scan": scan,
        "total_discovered": len(captures),
        "materialized_this_run": generated,
        "reused_complete_outputs": reused,
        "manually_reviewed_this_run": manual,
        "needs_review": sum(not row["accepted"] for row in rows),
        "manifest": str(manifest_path),
    }
    _atomic_json(output / "segmentation_run.json", report)
    return report

