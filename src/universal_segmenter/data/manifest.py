"""Read, write, validate, and discover capture-manifest records."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA_VERSION = "universal-paper-capture-manifest-1.0"
REQUIRED_FIELDS = (
    "capture_id",
    "document_id",
    "page_id",
    "fragment_id",
    "capture_type",
    "device_id",
    "capture_session_id",
    "background_family",
    "lighting_family",
    "image_path",
    "source_sha256",
)
VALID_CAPTURE_TYPES = {"scanner", "phone", "synthetic", "negative"}
VALID_ANNOTATION_STATES = {"unlabeled", "in_progress", "reviewed", "rejected"}
CAPTURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_capture_id(path: Path) -> str:
    # Content alone defines identity. This makes independently renamed copies
    # converge to one filename when contributors merge their dataset folders.
    return f"capture_{sha256_file(path)}"


def _safe_label(value: str, fallback: str) -> str:
    label = "".join(character if character.isalnum() else "_" for character in value)
    label = "_".join(part for part in label.split("_") if part)
    return label.lower() or fallback


def _infer_capture_type(relative_path: Path) -> str:
    tokens = {
        _safe_label(part, "")
        for part in relative_path.parts
        if part
    }
    phone_words = ("phone", "camera", "photo", "mobile")
    scanner_words = ("scanner", "printer", "flatbed", "scan_printer")
    if any(word in token for token in tokens for word in scanner_words):
        return "scanner"
    if any(word in token for token in tokens for word in phone_words):
        return "phone"
    if relative_path.suffix.lower() == ".pdf":
        return "scanner"
    return "phone"


def _is_capture_path(relative_path: Path) -> bool:
    if relative_path.suffix.lower() != ".pdf":
        return True
    capture_type = _infer_capture_type(relative_path)
    # A PDF stored beside phone lighting folders is normally the clean source
    # document, not a camera capture that needs a mask. Scanner/printer PDFs
    # remain valid annotation inputs.
    return capture_type == "scanner"


def _page_folder(containing_folder: Path) -> Path:
    """Collapse lighting/session subfolders under an explicit page folder."""
    parts = containing_folder.parts
    for index, part in enumerate(parts):
        normalized = _safe_label(part, "")
        if re.fullmatch(r"(?:pdf|page)_?\d+", normalized):
            return Path(*parts[:index + 1])
    return containing_folder


def capture_output_relative_path(record: dict[str, Any], suffix: str = ".png") -> Path:
    """Return a safe page-grouped output path for one capture record."""
    if "output_group" in record:
        raw_group = str(record.get("output_group", "captures"))
    else:
        raw_group = str(
            record.get("capture_folder") or record.get("page_folder", ""))
    raw_group = raw_group.replace("\\", "/")
    group_parts = [
        _safe_label(part, "page")
        for part in raw_group.split("/")
        if part not in ("", ".", "..")
    ]
    if not group_parts:
        group_parts = [_safe_label(str(record.get("page_id", "page")), "page")]
    filename = f"{_safe_label(str(record['capture_id']), 'capture')}{suffix}"
    return Path(*group_parts) / filename


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if not str(record.get(field, "")).strip():
            errors.append(f"missing required field: {field}")
    if record.get("capture_type") not in VALID_CAPTURE_TYPES:
        errors.append(f"invalid capture_type: {record.get('capture_type')!r}")
    state = record.get("annotation_status", "unlabeled")
    if state not in VALID_ANNOTATION_STATES:
        errors.append(f"invalid annotation_status: {state!r}")
    if state == "reviewed" and not str(record.get("mask_path", "")).strip():
        errors.append("reviewed capture has no mask_path")
    return errors


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not Path(path).is_file():
        return records
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def discover_captures(input_folder: Path) -> list[dict[str, Any]]:
    """Recursively discover captures and infer page metadata from folders.

    Every directory containing supported files is one page group. The complete
    path relative to ``input_folder`` becomes its stable page identifier.
    """
    input_folder = Path(input_folder)
    paths = sorted(
        path for path in input_folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in CAPTURE_EXTENSIONS
        and _is_capture_path(path.relative_to(input_folder))
    )
    records = []
    for path in paths:
        relative = path.relative_to(input_folder)
        parent = relative.parent
        containing_folder = (
            parent if parent != Path(".") else Path(input_folder.name)
        )
        page_folder = _page_folder(containing_folder)
        folder_text = page_folder.as_posix()
        capture_folder = containing_folder.as_posix()
        page_id = _safe_label(folder_text, "page")
        page_parts = page_folder.parts
        document_source = (
            page_parts[-1] if page_parts
            else input_folder.name
        )
        document_id = _safe_label(document_source, "document")
        capture_type = _infer_capture_type(relative)
        fragment_id = _safe_label(path.stem, "piece")
        records.append({
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "capture_id": safe_capture_id(path),
            "document_id": document_id,
            "page_id": page_id,
            "page_folder": folder_text,
            "capture_folder": capture_folder,
            "fragment_id": fragment_id,
            "capture_type": capture_type,
            "device_id": f"auto_{capture_type}",
            "capture_session_id": _safe_label(capture_folder, page_id),
            "background_family": (
                "scanner_bed_auto" if capture_type == "scanner"
                else "real_background_auto"
            ),
            "background_asset_id": "",
            "lighting_family": _safe_label(containing_folder.name, "default"),
            "fragment_size_group": "unmeasured",
            "dark_edge_categories": [],
            "challenge_tags": [],
            "image_path": path.as_posix(),
            "source_sha256": sha256_file(path),
            "dpi": None,
            "annotation_status": "unlabeled",
            "mask_path": "",
            "annotation_version": 0,
            "notes": "",
        })
    return records


def discover_capture_page_folders(input_folder: Path) -> list[str]:
    """Return page-policy folder names without hashing or decoding captures."""
    input_folder = Path(input_folder)
    folders: set[str] = set()
    for path in input_folder.rglob("*"):
        if (not path.is_file()
                or path.suffix.lower() not in CAPTURE_EXTENSIONS):
            continue
        relative = path.relative_to(input_folder)
        if not _is_capture_path(relative):
            continue
        parent = relative.parent
        containing = parent if parent != Path(".") else Path(input_folder.name)
        folders.add(_page_folder(containing).as_posix())
    return sorted(folders)


def sync_capture_manifest(input_folder: Path, manifest_path: Path,
                          project: Path | None = None) -> dict[str, int]:
    """Merge a recursive folder scan into a resumable capture manifest."""
    input_folder = Path(input_folder)
    manifest_path = Path(manifest_path)
    project = Path(project).resolve() if project is not None else None
    input_folder.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(manifest_path)
    by_hash = {str(row.get("source_sha256", "")): row for row in existing}
    discovered = discover_captures(input_folder)
    added = updated = 0

    for automatic in discovered:
        source = Path(automatic["image_path"])
        if project is not None:
            try:
                automatic["image_path"] = source.resolve().relative_to(project).as_posix()
            except ValueError:
                automatic["image_path"] = source.resolve().as_posix()
        previous = by_hash.get(automatic["source_sha256"])
        if previous is None:
            existing.append(automatic)
            by_hash[automatic["source_sha256"]] = automatic
            added += 1
            continue

        changed = False
        for field, value in automatic.items():
            if field in {"annotation_status", "annotation_version", "mask_path", "notes"}:
                continue
            current = previous.get(field)
            if field in {"image_path", "page_folder", "capture_folder"} or current in (
                    None, "", "TODO"):
                if current != value:
                    previous[field] = value
                    changed = True
        updated += int(changed)

    existing.sort(key=lambda row: (
        str(row.get("page_id", "")), str(row.get("fragment_id", "")),
        str(row.get("capture_id", ""))))
    write_jsonl(manifest_path, existing)
    return {
        "discovered": len(discovered),
        "added": added,
        "updated": updated,
        "total": len(existing),
    }
