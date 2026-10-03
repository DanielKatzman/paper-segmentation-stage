"""Freeze a generated dataset before any model-selection run begins."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FREEZE_FILENAME = "dataset_freeze.json"
REQUIRED_SPLITS = ("train", "validation", "calibration", "holdout")
REQUIRED_COLUMNS = {
    "sample_id", "split", "sample_kind", "capture_profile", "image_path",
    "mask_path", "group_id",
}


def read_sample_rows(dataset_root: Path) -> list[dict[str, str]]:
    path = Path(dataset_root) / "samples.csv"
    if not path.is_file():
        raise FileNotFoundError(f"generated sample manifest is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"samples.csv is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("samples.csv contains no samples")
    return rows


def _safe_dataset_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"sample path escapes dataset root: {relative}")
    return candidate


def validate_dataset_rows(dataset_root: Path,
                          rows: list[dict[str, str]]) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    report_path = root / "dataset_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"dataset report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("valid") or report.get("errors"):
        raise ValueError(f"dataset_report.json is not valid: {report.get('errors', [])[:8]}")
    if int(report.get("sample_count", -1)) != len(rows):
        raise ValueError("dataset report/sample manifest count mismatch")

    sample_ids = [row["sample_id"] for row in rows]
    duplicates = [name for name, count in Counter(sample_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate sample IDs: {duplicates[:8]}")
    split_counts = Counter(row["split"] for row in rows)
    real_split_counts = Counter(
        row["split"] for row in rows if row["sample_kind"] == "real_capture")
    missing_splits = [name for name in REQUIRED_SPLITS if real_split_counts[name] == 0]
    if missing_splits:
        raise ValueError(
            "cannot freeze dataset: untouched real captures are missing from "
            f"required splits {missing_splits}. Repair the versioned split "
            "manifest, remove only the incomplete generated dataset version, "
            "and regenerate it.")

    kinds = Counter(row["sample_kind"] for row in rows)
    profiles = Counter(row["capture_profile"] for row in rows)
    split_profiles: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        image = _safe_dataset_file(root, row["image_path"])
        mask = _safe_dataset_file(root, row["mask_path"])
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(
                f"sample {row['sample_id']} is missing image or mask")
        if row["sample_kind"] == "real_capture":
            split_profiles[row["split"]][row["capture_profile"]] += 1
        elif row["split"] != "train":
            raise ValueError(
                f"generated sample {row['sample_id']} leaked into {row['split']}")
    return {
        "sample_count": len(rows),
        "samples_by_split": dict(split_counts),
        "real_captures_by_split": dict(real_split_counts),
        "real_profiles_by_split": {
            split: dict(values) for split, values in sorted(split_profiles.items())},
        "samples_by_kind": dict(kinds),
        "samples_by_profile": dict(profiles),
    }


def dataset_fingerprint(dataset_root: Path,
                        rows: list[dict[str, str]] | None = None) -> str:
    """Fast manifest/stat fingerprint used to reject changed resume data."""
    root = Path(dataset_root).resolve()
    rows = rows or read_sample_rows(root)
    digest = hashlib.sha256()
    for name in ("samples.csv", "samples.jsonl", "dataset_report.json"):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        digest.update(row["sample_id"].encode("utf-8"))
        for key in ("image_path", "mask_path"):
            path = _safe_dataset_file(root, row[key])
            stat = path.stat()
            digest.update(row[key].encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def freeze_dataset(dataset_root: Path) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    rows = read_sample_rows(root)
    summary = validate_dataset_rows(root, rows)
    frozen = {
        "schema_version": "universal-paper-dataset-freeze-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint": dataset_fingerprint(root, rows),
        **summary,
    }
    path = root / FREEZE_FILENAME
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return frozen


def load_dataset_freeze(dataset_root: Path) -> dict[str, Any]:
    path = Path(dataset_root).resolve() / FREEZE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"dataset is not frozen: {path}. Run the matching versioned "
            "freeze_dataset script first.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "universal-paper-dataset-freeze-1.0":
        raise ValueError(f"unsupported dataset freeze schema: {path}")
    return value


def verify_dataset_freeze(dataset_root: Path) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    frozen = load_dataset_freeze(root)
    rows = read_sample_rows(root)
    validate_dataset_rows(root, rows)
    current = dataset_fingerprint(root, rows)
    if current != frozen.get("fingerprint"):
        raise ValueError(
            "frozen dataset changed after approval. Restore the frozen files or "
            "create a new dataset version and experiment run.")
    return frozen
