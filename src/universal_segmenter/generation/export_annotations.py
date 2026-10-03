"""Export reviewed annotation-manifest rows into generator real-piece inputs."""

from __future__ import annotations

import csv
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from universal_segmenter.data.manifest import (
    capture_output_relative_path,
    read_jsonl,
    write_jsonl,
)
from universal_segmenter.data.physical_identity import (
    assign_physical_piece_identity,
)
from universal_segmenter.data.splits import (
    SPLITS,
    assign_grouped_splits,
    leakage_errors,
    required_split_errors,
)

from .sources import IMAGE_EXTENSIONS, VALID_CAPTURE_TYPES, VALID_SPLITS, create_source_layout


DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.10,
    "calibration": 0.10,
    "holdout": 0.10,
}


def _assign_physical_piece_ids(prepared: list[dict[str, Any]]) -> str:
    """Compatibility wrapper around the shared identity implementation."""
    records = [item["record"] for item in prepared]
    method = assign_physical_piece_identity(records)
    for item in prepared:
        item["physical_piece_id"] = str(
            item["record"]["physical_piece_id"])
    return method


def _resolve(project: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project / path


def _copy_if_changed(source: Path, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        source_stat, destination_stat = source.stat(), destination.stat()
        if (source_stat.st_size == destination_stat.st_size
                and source_stat.st_mtime_ns == destination_stat.st_mtime_ns):
            return False
    temporary = destination.with_suffix(destination.suffix + ".export.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    return True


def _atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "image_filename", "mask_filename", "piece_id", "document_id",
        "page_id", "capture_session_id", "capture_type", "split",
        "canonical_for_compositing",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _load_or_create_splits(
        reviewed: list[dict[str, Any]], split_manifest: Path,
        split_ratios: dict[str, float], split_seed: int,
        ) -> list[dict[str, str]]:
    assignments = read_jsonl(split_manifest)
    reviewed_ids = {str(row["capture_id"]) for row in reviewed}
    if assignments:
        by_capture = {str(row.get("capture_id", "")): row for row in assignments}
        missing = sorted(reviewed_ids - set(by_capture))
        if missing:
            raise ValueError(
                f"Split manifest is missing {len(missing)} reviewed captures. "
                f"Delete or update {split_manifest}, then export again. "
                f"First missing IDs: {missing[:5]}")
        selected = [by_capture[capture_id] for capture_id in sorted(reviewed_ids)]
    else:
        selected = assign_grouped_splits(reviewed, split_ratios, split_seed)
        for row in selected:
            row["schema_version"] = "universal-paper-split-manifest-1.0"
        write_jsonl(split_manifest, selected)
    invalid = sorted({
        str(row.get("split", "")) for row in selected
        if str(row.get("split", "")) not in VALID_SPLITS
    })
    if invalid:
        raise ValueError(f"Unsupported split names in {split_manifest}: {invalid}")
    errors = leakage_errors(reviewed, selected)
    if errors:
        raise ValueError("Split leakage detected: " + "; ".join(errors[:8]))
    errors = required_split_errors(selected, (
        name for name in SPLITS if split_ratios.get(name, 0.0) > 0.0))
    if errors:
        raise ValueError(
            "Split manifest cannot be frozen: " + "; ".join(errors) +
            f". Run scripts/assign_splits.py to rewrite {split_manifest}, "
            "then export again.")
    return selected


def export_reviewed_annotations(
        project_root: Path,
        capture_manifest: Path = Path("manifests/captures.jsonl"),
        split_manifest: Path = Path("manifests/splits.jsonl"),
        generator_input_root: Path = Path("data/generator_inputs"),
        split_ratios: dict[str, float] | None = None,
        split_seed: int = 64319,
        require_complete: bool = True,
        ) -> dict[str, Any]:
    """Materialize reviewed image/mask pairs and their leakage-safe CSV."""
    project = Path(project_root).resolve()
    capture_path = _resolve(project, str(capture_manifest))
    split_path = _resolve(project, str(split_manifest))
    generator_root = _resolve(project, str(generator_input_root))
    records = read_jsonl(capture_path)
    if not records:
        raise RuntimeError(f"Capture manifest is missing or empty: {capture_path}")

    unfinished = [
        str(row.get("capture_id", "?")) for row in records
        if row.get("annotation_status") not in {"reviewed", "rejected"}
    ]
    if require_complete and unfinished:
        raise RuntimeError(
            f"Cannot export while {len(unfinished)} captures are not reviewed. "
            f"First unfinished IDs: {unfinished[:8]}")
    reviewed = [
        row for row in records if row.get("annotation_status") == "reviewed"
    ]
    if not reviewed:
        raise RuntimeError("Capture manifest has no reviewed annotations to export")

    assignments = _load_or_create_splits(
        reviewed, split_path, split_ratios or DEFAULT_SPLIT_RATIOS, split_seed)
    split_by_capture = {
        str(row["capture_id"]): str(row["split"]) for row in assignments
    }
    source_layout = create_source_layout(generator_root)
    real_root = source_layout["real_pieces"]

    prepared: list[dict[str, Any]] = []
    destination_keys: set[tuple[str, str]] = set()
    for record in sorted(reviewed, key=lambda row: str(row["capture_id"])):
        capture_id = str(record["capture_id"])
        capture_type = str(record.get("capture_type", "")).lower()
        if capture_type not in VALID_CAPTURE_TYPES:
            raise ValueError(
                f"Reviewed capture {capture_id} has invalid capture_type "
                f"{capture_type!r}")
        image_value = str(
            record.get("annotation_image_path") or record.get("image_path") or ""
        ).strip()
        mask_value = str(record.get("mask_path", "")).strip()
        if not image_value or not mask_value:
            raise ValueError(
                f"Reviewed capture {capture_id} is missing image or mask metadata")
        source_image = _resolve(project, image_value)
        source_mask = _resolve(project, mask_value)
        if not source_image.is_file() or not source_mask.is_file():
            raise ValueError(
                f"Reviewed capture {capture_id} references a missing image or mask: "
                f"{source_image}, {source_mask}")
        if source_image.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Capture {capture_id} needs a materialized image, not "
                f"{source_image.suffix}: {source_image}")
        with Image.open(source_image) as handle:
            image_size = ImageOps.exif_transpose(handle).size
        with Image.open(source_mask) as handle:
            mask_size = ImageOps.exif_transpose(handle).size
        if image_size != mask_size:
            raise ValueError(
                f"Annotation image/mask dimensions differ for {capture_id}: "
                f"{image_size} versus {mask_size}")

        base = capture_output_relative_path(record)
        image_relative = base.with_suffix(source_image.suffix.lower())
        mask_relative = base.with_suffix(".png")
        key = (image_relative.as_posix(), mask_relative.as_posix())
        if key in destination_keys:
            raise ValueError(f"Duplicate exported destination for {capture_id}: {key}")
        destination_keys.add(key)
        prepared.append({
            "record": record,
            "source_image": source_image,
            "source_mask": source_mask,
            "image_relative": image_relative,
            "mask_relative": mask_relative,
            "split": split_by_capture[capture_id],
        })

    physical_identity_method = _assign_physical_piece_ids(prepared)

    by_piece: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in prepared:
        record = item["record"]
        key = (
            str(record["document_id"]), str(record["page_id"]),
            str(item["physical_piece_id"]),
        )
        by_piece.setdefault(key, []).append(item)
    canonical_ids: set[str] = set()
    for group in by_piece.values():
        training = [item for item in group if item["split"] == "train"]
        if training:
            canonical_ids.add(min(
                str(item["record"]["capture_id"]) for item in training))

    rows: list[dict[str, str]] = []
    copied = reused = 0
    for item in prepared:
        record = item["record"]
        image_destination = real_root / "images" / item["image_relative"]
        mask_destination = real_root / "masks" / item["mask_relative"]
        image_copied = _copy_if_changed(item["source_image"], image_destination)
        mask_copied = _copy_if_changed(item["source_mask"], mask_destination)
        copied += int(image_copied) + int(mask_copied)
        reused += 2 - int(image_copied) - int(mask_copied)
        capture_id = str(record["capture_id"])
        rows.append({
            "image_filename": item["image_relative"].as_posix(),
            "mask_filename": item["mask_relative"].as_posix(),
            "piece_id": str(item["physical_piece_id"]),
            "document_id": str(record["document_id"]),
            "page_id": str(record["page_id"]),
            "capture_session_id": str(record["capture_session_id"]),
            "capture_type": str(record["capture_type"]),
            "split": str(item["split"]),
            "canonical_for_compositing": (
                "true" if capture_id in canonical_ids else "false"),
        })
    _atomic_csv(real_root / "pieces.csv", rows)
    return {
        "exported_captures": len(rows),
        "physical_pieces": len(by_piece),
        "physical_identity_method": physical_identity_method,
        "canonical_training_pieces": len(canonical_ids),
        "copied_files": copied,
        "reused_files": reused,
        "splits": dict(Counter(row["split"] for row in rows)),
        "pieces_csv": str(real_root / "pieces.csv"),
        "split_manifest": str(split_path),
    }
