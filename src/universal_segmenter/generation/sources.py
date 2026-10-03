"""Discover and validate the generator's three independent source collections."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
VALID_SPLITS = {"train", "validation", "calibration", "holdout"}
VALID_CAPTURE_TYPES = {"phone", "scanner"}
BACKGROUND_FIELDS = ("filename", "capture_type", "session_id")
PDF_FIELDS = ("filename", "document_id", "license")
PIECE_FIELDS = (
    "image_filename", "mask_filename", "piece_id", "document_id",
    "page_id", "capture_session_id", "capture_type", "split",
    "canonical_for_compositing",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(*parts: object, length: int = 16) -> str:
    value = "|".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class BackgroundSource:
    asset_id: str
    path: Path
    capture_type: str
    session_id: str
    sha256: str


@dataclass(frozen=True)
class PdfSource:
    document_id: str
    path: Path
    sha256: str
    license_name: str = ""


@dataclass(frozen=True)
class RealPieceSource:
    piece_id: str
    document_id: str
    page_id: str
    capture_session_id: str
    capture_type: str
    split: str
    image_path: Path
    mask_path: Path
    canonical_for_compositing: bool
    image_sha256: str
    mask_sha256: str


def create_source_layout(root: Path) -> dict[str, Path]:
    """Create the requested three-folder input contract without overwriting data."""
    root = Path(root)
    backgrounds = root / "backgrounds"
    pdfs = root / "pdfs"
    real_pieces = root / "real_pieces"
    for folder in (
        backgrounds / "phone",
        backgrounds / "scanner",
        pdfs,
        real_pieces / "images",
        real_pieces / "masks",
        real_pieces / "manifests",
    ):
        folder.mkdir(parents=True, exist_ok=True)

    backgrounds_csv = backgrounds / "backgrounds.csv"
    if not backgrounds_csv.exists():
        backgrounds_csv.write_text(
            "filename,capture_type,session_id\n",
            encoding="utf-8",
        )
    pdfs_csv = pdfs / "pdfs.csv"
    if not pdfs_csv.exists():
        pdfs_csv.write_text(
            "filename,document_id,license\n",
            encoding="utf-8",
        )
    pieces_csv = real_pieces / "pieces.csv"
    if not pieces_csv.exists():
        pieces_csv.write_text(
            "image_filename,mask_filename,piece_id,document_id,page_id,"
            "capture_session_id,capture_type,split,canonical_for_compositing\n",
            encoding="utf-8",
        )
    return {"backgrounds": backgrounds, "pdfs": pdfs, "real_pieces": real_pieces}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(path: Path, fieldnames: tuple[str, ...],
                rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _infer_capture_type(path: Path, default: str = "phone") -> str:
    text = "/".join(part.casefold() for part in path.parts)
    if any(token in text for token in (
            "scanner", "scan_printer", "printer_scan", "flatbed")):
        return "scanner"
    if any(token in text for token in (
            "phone", "photo", "camera", "mobile")):
        return "phone"
    return default


def _file_stem_key(relative: Path) -> str:
    return relative.with_suffix("").as_posix().casefold()


def _indexed_images(folder: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in sorted(Path(folder).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(folder)
        key = _file_stem_key(relative)
        if key in output:
            raise ValueError(
                f"multiple files have the same relative stem under {folder}: "
                f"{output[key].name}, {path.name}")
        output[key] = path
    return output


def _v1_real_capture_metadata(dataset_root: Path | None) -> dict[
        tuple[str, str], dict[str, Any]]:
    if dataset_root is None:
        return {}
    samples_path = Path(dataset_root) / "samples.jsonl"
    if not samples_path.is_file():
        return {}
    output: dict[tuple[str, str], dict[str, Any]] = {}
    with samples_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sample_kind") != "real_capture":
                continue
            lineage = row.get("lineage") or {}
            image_hash = str(lineage.get("source_image_sha256", "")).strip()
            mask_hash = str(lineage.get("source_mask_sha256", "")).strip()
            if not image_hash or not mask_hash:
                raise ValueError(
                    f"{samples_path}:{line_number} real capture lacks source hashes")
            key = (image_hash, mask_hash)
            if key in output:
                raise ValueError(
                    "v1 dataset contains duplicate real source hash pairs")
            output[key] = row
    return output


def _annotation_capture_metadata(real_root: Path) -> dict[str, dict[str, str]]:
    """Load merge-safe annotation manifests copied beside images and masks."""
    output: dict[str, dict[str, str]] = {}
    manifest_root = Path(real_root) / "manifests"
    if not manifest_root.is_dir():
        return output
    for path in sorted(manifest_root.rglob("captures_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("annotation_status") != "reviewed":
                    continue
                capture_id = str(record.get("capture_id", "")).strip()
                split = str(record.get("dataset_split", "")).strip().lower()
                piece_id = str(record.get("physical_piece_id", "")).strip()
                if not capture_id or not split or not piece_id:
                    continue
                if split not in VALID_SPLITS:
                    raise ValueError(
                        f"{path}:{line_number} has invalid dataset_split "
                        f"{split!r}")
                metadata = {
                    "piece_id": piece_id,
                    "document_id": str(record.get("document_id", "")).strip(),
                    "page_id": str(record.get("page_id", "")).strip(),
                    "capture_session_id": str(
                        record.get("capture_session_id", "")).strip(),
                    "capture_type": str(record.get("capture_type", "")).strip().lower(),
                    "split": split,
                }
                missing = [name for name, value in metadata.items() if not value]
                if missing:
                    raise ValueError(
                        f"{path}:{line_number} is missing annotation split "
                        f"metadata {missing}")
                previous = output.get(capture_id)
                if previous is not None and previous != metadata:
                    raise ValueError(
                        f"conflicting annotation manifests for {capture_id}: "
                        f"{previous} versus {metadata}")
                output[capture_id] = metadata
    return output


def sync_generator_inputs(
        root: Path, baseline_dataset_root: Path | None = None,
        ) -> dict[str, Any]:
    """Scan all three input folders and rebuild their metadata inventories.

    Annotator manifests are the source of truth for split and physical-piece
    metadata. Existing CSV rows remain supported. New loose image/mask pairs
    without a manifest are deliberately assigned to training only: without
    identity metadata they must not silently enter evaluation splits.
    """
    layout = create_source_layout(root)

    background_root = layout["backgrounds"]
    old_backgrounds = _read_csv(background_root / "backgrounds.csv")
    background_metadata = {
        str(row.get("filename", "")).replace("\\", "/"): row
        for row in old_backgrounds if str(row.get("filename", "")).strip()
    }
    background_rows: list[dict[str, str]] = []
    for path in sorted(background_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(background_root).as_posix()
        previous = background_metadata.get(relative) or background_metadata.get(
            path.name, {})
        digest = sha256_file(path)
        capture_type = str(previous.get("capture_type", "")).strip().lower()
        if capture_type not in VALID_CAPTURE_TYPES:
            capture_type = _infer_capture_type(Path(relative))
        background_rows.append({
            "filename": relative,
            "capture_type": capture_type,
            "session_id": str(previous.get("session_id", "")).strip()
            or f"background_{digest[:12]}",
        })
    _atomic_csv(
        background_root / "backgrounds.csv", BACKGROUND_FIELDS,
        background_rows)

    pdf_root = layout["pdfs"]
    old_pdfs = _read_csv(pdf_root / "pdfs.csv")
    pdf_metadata = {
        str(row.get("filename", "")).replace("\\", "/"): row
        for row in old_pdfs if str(row.get("filename", "")).strip()
    }
    pdf_rows: list[dict[str, str]] = []
    for path in sorted(pdf_root.rglob("*.pdf")):
        relative = path.relative_to(pdf_root).as_posix()
        previous = pdf_metadata.get(relative) or pdf_metadata.get(path.name, {})
        digest = sha256_file(path)
        pdf_rows.append({
            "filename": relative,
            "document_id": str(previous.get("document_id", "")).strip()
            or f"pdf_{digest[:16]}",
            "license": str(previous.get("license", "")).strip(),
        })
    _atomic_csv(pdf_root / "pdfs.csv", PDF_FIELDS, pdf_rows)

    real_root = layout["real_pieces"]
    image_root, mask_root = real_root / "images", real_root / "masks"
    images, masks = _indexed_images(image_root), _indexed_images(mask_root)
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        messages = []
        if missing_masks:
            messages.append(
                f"{len(missing_masks)} image(s) have no matching mask: "
                f"{missing_masks[:8]}")
        if missing_images:
            messages.append(
                f"{len(missing_images)} mask(s) have no matching image: "
                f"{missing_images[:8]}")
        raise ValueError(
            "Real-piece folders must contain matching relative filenames. "
            + "; ".join(messages))

    old_piece_rows = _read_csv(real_root / "pieces.csv")
    baseline_by_hash = _v1_real_capture_metadata(baseline_dataset_root)
    annotation_by_capture = _annotation_capture_metadata(real_root)
    old_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for row in old_piece_rows:
        image_name = str(row.get("image_filename", "")).replace("\\", "/")
        mask_name = str(row.get("mask_filename", "")).replace("\\", "/")
        if image_name and mask_name:
            old_by_pair[(
                _file_stem_key(Path(image_name)),
                _file_stem_key(Path(mask_name)),
            )] = row

    piece_rows: list[dict[str, str]] = []
    added = preserved = recovered_from_v1 = recovered_from_annotations = 0
    for key in sorted(images):
        image_path, mask_path = images[key], masks[key]
        image_relative = image_path.relative_to(image_root).as_posix()
        mask_relative = mask_path.relative_to(mask_root).as_posix()
        previous = old_by_pair.get((key, key))
        image_hash = sha256_file(image_path)
        mask_hash = sha256_file(mask_path)
        baseline = baseline_by_hash.get((image_hash, mask_hash))
        capture_id = Path(image_relative).stem
        annotation = annotation_by_capture.get(capture_id)
        previous_is_automatic = (
            previous is not None
            and str(previous.get("piece_id", "")).startswith("auto_"))
        if annotation is not None:
            row = {
                "image_filename": image_relative,
                "mask_filename": mask_relative,
                **annotation,
                "canonical_for_compositing": "false",
            }
            preserved += int(previous is not None)
            recovered_from_annotations += 1
        elif baseline is not None and (previous is None or previous_is_automatic):
            lineage = baseline.get("lineage") or {}
            row = {
                "image_filename": image_relative,
                "mask_filename": mask_relative,
                "piece_id": str(baseline.get("piece_id", "")).strip(),
                "document_id": str(baseline.get("document_id", "")).strip(),
                "page_id": str(baseline.get("page_id", "")).strip(),
                "capture_session_id": str(
                    lineage.get("capture_session_id", "")).strip(),
                "capture_type": str(
                    baseline.get("capture_profile", "")).strip().lower(),
                "split": str(baseline.get("split", "")).strip().lower(),
                "canonical_for_compositing": "false",
            }
            missing = [
                field for field in (
                    "piece_id", "document_id", "page_id",
                    "capture_session_id", "capture_type", "split")
                if not row[field]
            ]
            if missing:
                raise ValueError(
                    f"v1 real-capture metadata is missing {missing} for "
                    f"{image_relative}")
            preserved += 1
            recovered_from_v1 += 1
        elif previous is not None and not previous_is_automatic:
            row = {
                field: str(previous.get(field, "")).strip()
                for field in PIECE_FIELDS
            }
            row["image_filename"] = image_relative
            row["mask_filename"] = mask_relative
            preserved += 1
        else:
            automatic_id = f"auto_{image_hash[:16]}"
            row = {
                "image_filename": image_relative,
                "mask_filename": mask_relative,
                "piece_id": automatic_id,
                "document_id": automatic_id,
                "page_id": automatic_id,
                "capture_session_id": automatic_id,
                "capture_type": _infer_capture_type(Path(image_relative)),
                "split": "train",
                "canonical_for_compositing": "true",
            }
            added += 1
        piece_rows.append(row)

    # Recovered v1 samples do not carry the old CSV's canonical flag. Select
    # exactly one deterministic training capture per physical piece so the
    # composite generator retains its original behavior.
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in piece_rows:
        if row["split"] != "train":
            continue
        group_key = (row["document_id"], row["page_id"], row["piece_id"])
        groups.setdefault(group_key, []).append(row)
    for group in groups.values():
        if not any(
                row["canonical_for_compositing"].casefold()
                in {"1", "true", "yes", "y"} for row in group):
            min(group, key=lambda row: row["image_filename"])[
                "canonical_for_compositing"] = "true"
    _atomic_csv(real_root / "pieces.csv", PIECE_FIELDS, piece_rows)

    return {
        "backgrounds": len(background_rows),
        "pdfs": len(pdf_rows),
        "real_capture_pairs": len(piece_rows),
        "preserved_real_rows": preserved,
        "recovered_from_v1_freeze": recovered_from_v1,
        "assigned_from_annotation_manifests": recovered_from_annotations,
        "new_real_pairs_added_to_train": added,
        "stale_real_rows_removed": len(
            set(old_by_pair) - {(key, key) for key in images}),
        "backgrounds_csv": str(background_root / "backgrounds.csv"),
        "pdfs_csv": str(pdf_root / "pdfs.csv"),
        "pieces_csv": str(real_root / "pieces.csv"),
    }


def discover_backgrounds(folder: Path) -> list[BackgroundSource]:
    folder = Path(folder)
    metadata = {
        row.get("filename", "").replace("\\", "/"): row
        for row in _read_csv(folder / "backgrounds.csv")
        if row.get("filename", "").strip()
    }
    paths = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    output: list[BackgroundSource] = []
    for path in paths:
        relative = path.relative_to(folder).as_posix()
        row = metadata.get(relative) or metadata.get(path.name) or {}
        inferred = "phone" if "phone" in path.parts else "scanner" if "scanner" in path.parts else ""
        capture_type = str(row.get("capture_type", inferred)).strip().lower()
        if capture_type not in VALID_CAPTURE_TYPES:
            raise ValueError(
                f"Background {relative} needs capture_type phone or scanner in backgrounds.csv"
            )
        digest = sha256_file(path)
        session_id = str(row.get("session_id", "")).strip() or f"background_{digest[:12]}"
        output.append(BackgroundSource(
            asset_id=f"bg_{digest[:16]}",
            path=path,
            capture_type=capture_type,
            session_id=session_id,
            sha256=digest,
        ))
    return output


def discover_pdfs(folder: Path) -> list[PdfSource]:
    folder = Path(folder)
    metadata = {
        row.get("filename", "").replace("\\", "/"): row
        for row in _read_csv(folder / "pdfs.csv")
        if row.get("filename", "").strip()
    }
    output: list[PdfSource] = []
    for path in sorted(folder.rglob("*.pdf")):
        relative = path.relative_to(folder).as_posix()
        row = metadata.get(relative) or metadata.get(path.name) or {}
        digest = sha256_file(path)
        document_id = str(row.get("document_id", "")).strip() or f"pdf_{digest[:16]}"
        output.append(PdfSource(
            document_id=document_id,
            path=path,
            sha256=digest,
            license_name=str(row.get("license", "")).strip(),
        ))
    return output


def discover_real_pieces(folder: Path) -> list[RealPieceSource]:
    """Read explicit image/mask pairs; never guess grouping metadata."""
    folder = Path(folder)
    rows = _read_csv(folder / "pieces.csv")
    output: list[RealPieceSource] = []
    seen_capture_files: set[tuple[str, str]] = set()
    for line_number, row in enumerate(rows, start=2):
        if not any(str(value).strip() for value in row.values()):
            continue
        required = (
            "image_filename", "mask_filename", "piece_id", "document_id",
            "page_id", "capture_session_id", "capture_type", "split",
        )
        missing = [name for name in required if not str(row.get(name, "")).strip()]
        if missing:
            raise ValueError(f"pieces.csv line {line_number} missing: {missing}")
        image_path = folder / "images" / str(row["image_filename"])
        mask_path = folder / "masks" / str(row["mask_filename"])
        if not image_path.is_file() or not mask_path.is_file():
            raise ValueError(
                f"pieces.csv line {line_number} references a missing image or mask"
            )
        capture_type = str(row["capture_type"]).strip().lower()
        split = str(row["split"]).strip().lower()
        if capture_type not in VALID_CAPTURE_TYPES:
            raise ValueError(f"pieces.csv line {line_number}: invalid capture_type")
        if split not in VALID_SPLITS:
            raise ValueError(f"pieces.csv line {line_number}: invalid split {split!r}")
        file_key = (str(image_path.resolve()), str(mask_path.resolve()))
        if file_key in seen_capture_files:
            raise ValueError(f"duplicate real capture row: {file_key}")
        seen_capture_files.add(file_key)
        piece_id = str(row["piece_id"]).strip()
        capture_session_id = str(row["capture_session_id"]).strip()
        canonical = str(row.get("canonical_for_compositing", "")).strip().lower()
        output.append(RealPieceSource(
            piece_id=piece_id,
            document_id=str(row["document_id"]).strip(),
            page_id=str(row["page_id"]).strip(),
            capture_session_id=capture_session_id,
            capture_type=capture_type,
            split=split,
            image_path=image_path,
            mask_path=mask_path,
            canonical_for_compositing=canonical in {"1", "true", "yes", "y"},
            image_sha256=sha256_file(image_path),
            mask_sha256=sha256_file(mask_path),
        ))

    by_piece: dict[tuple[str, str, str], list[RealPieceSource]] = {}
    for source in output:
        key = (source.document_id, source.page_id, source.piece_id)
        by_piece.setdefault(key, []).append(source)
    for piece_key, group in by_piece.items():
        group_splits = {source.split for source in group}
        if len(group_splits) != 1:
            raise ValueError(f"physical piece {piece_key!r} crosses splits: {group_splits}")
        page_keys = {(source.document_id, source.page_id) for source in group}
        if len(page_keys) != 1:
            raise ValueError(f"physical piece {piece_key!r} has conflicting page metadata")
        canonical_count = sum(source.canonical_for_compositing for source in group)
        if canonical_count > 1:
            raise ValueError(f"physical piece {piece_key!r} has multiple canonical cutouts")
    return output
