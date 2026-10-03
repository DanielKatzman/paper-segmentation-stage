"""Compile real and synthetic sources into one materialized segmentation dataset."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

from .capture import (
    CaptureRenderError,
    load_real_piece,
    piece_from_page,
    render_background_only,
    render_capture,
    retexture_real_shape,
)
from .geometry import GeometryError, Partition, generate_partition, rasterize_partition
from .sources import (
    BackgroundSource,
    PdfSource,
    RealPieceSource,
    create_source_layout,
    discover_backgrounds,
    discover_pdfs,
    discover_real_pieces,
    stable_id,
)


DATASET_SCHEMA_VERSION = "universal-paper-generated-dataset-1.0"


@dataclass(frozen=True)
class GeneratorConfig:
    input_root: Path = Path("data/generator_inputs")
    output_root: Path = Path("data/complete_dataset_v1")
    random_seed: int = 73191
    image_size: int = 768
    pdf_dpi: int = 120
    max_pages_per_pdf: int | None = 5
    puzzles_per_pdf_page: int = 2
    min_pieces_per_puzzle: int = 4
    max_pieces_per_puzzle: int = 8
    geometry_raster_attempts: int = 8
    puzzle_render_attempts: int = 8
    capture_pose_attempts: int = 24
    variants_per_synthetic_piece: int = 1
    real_composites_per_piece: int = 12
    real_retextures_per_piece: int = 4
    background_negative_ratio: float = 0.05
    preview_count: int = 30
    require_real_backgrounds: bool = True
    jpeg_quality: int = 95
    show_progress: bool = True
    progress_update_seconds: float = 1.0


def _seed(base: int, *parts: object) -> int:
    return int(stable_id(base, *parts, length=16), 16) & 0x7FFF_FFFF_FFFF_FFFF


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class _ProgressReporter:
    """Small dependency-free progress display suitable for PyCharm consoles."""

    def __init__(self, enabled: bool, update_seconds: float) -> None:
        self.enabled = bool(enabled)
        self.update_seconds = float(update_seconds)
        self.started = monotonic()
        self.phase_name = ""
        self.phase_total = 0
        self.phase_completed = 0
        self.phase_started = self.started
        self.last_update = self.started

    def note(self, message: str) -> None:
        if self.enabled:
            print(f"[dataset] {message}", flush=True)

    def phase(self, name: str, total: int) -> None:
        self.phase_name = name
        self.phase_total = max(0, int(total))
        self.phase_completed = 0
        self.phase_started = monotonic()
        self.last_update = self.phase_started
        if self.phase_total:
            self.note(f"{name}: 0/{self.phase_total:,} (0.0%)")
        else:
            self.note(f"{name}: skipped (0 items)")

    def advance(self, detail: str = "") -> None:
        if self.phase_total <= 0:
            return
        self.phase_completed = min(self.phase_completed + 1, self.phase_total)
        now = monotonic()
        finished = self.phase_completed == self.phase_total
        first = self.phase_completed == 1
        if not first and not finished and now - self.last_update < self.update_seconds:
            return
        elapsed = max(now - self.phase_started, 1.0e-9)
        rate = self.phase_completed / elapsed
        remaining = self.phase_total - self.phase_completed
        eta = remaining / rate if rate > 0.0 else 0.0
        percent = 100.0 * self.phase_completed / self.phase_total
        message = (
            f"{self.phase_name}: {self.phase_completed:,}/{self.phase_total:,} "
            f"({percent:5.1f}%) | elapsed {_format_duration(elapsed)}"
        )
        if not finished:
            message += f" | ETA {_format_duration(eta)}"
        if detail:
            message += f" | {detail}"
        self.note(message)
        self.last_update = now

    def finish(self, sample_count: int) -> None:
        self.note(
            f"Complete: {sample_count:,} samples in "
            f"{_format_duration(monotonic() - self.started)}")


def _validate_config(config: GeneratorConfig) -> None:
    if config.random_seed < 0:
        raise ValueError("random_seed must be non-negative")
    if config.image_size < 128:
        raise ValueError("image_size must be at least 128")
    if config.pdf_dpi < 36 or config.pdf_dpi > 600:
        raise ValueError("pdf_dpi must be between 36 and 600")
    if config.max_pages_per_pdf is not None and config.max_pages_per_pdf < 1:
        raise ValueError("max_pages_per_pdf must be None or at least 1")
    if config.puzzles_per_pdf_page < 1:
        raise ValueError("puzzles_per_pdf_page must be at least 1")
    if config.min_pieces_per_puzzle < 2:
        raise ValueError("min_pieces_per_puzzle must be at least 2")
    if config.min_pieces_per_puzzle > config.max_pieces_per_puzzle:
        raise ValueError("min_pieces_per_puzzle exceeds max_pieces_per_puzzle")
    if config.geometry_raster_attempts < 1:
        raise ValueError("geometry_raster_attempts must be at least 1")
    if config.puzzle_render_attempts < 1:
        raise ValueError("puzzle_render_attempts must be at least 1")
    if config.capture_pose_attempts < 1:
        raise ValueError("capture_pose_attempts must be at least 1")
    if config.variants_per_synthetic_piece < 1:
        raise ValueError("variants_per_synthetic_piece must be at least 1")
    if config.real_composites_per_piece < 0 or config.real_retextures_per_piece < 0:
        raise ValueError("real composite/retexture counts cannot be negative")
    if config.background_negative_ratio < 0.0:
        raise ValueError("background_negative_ratio cannot be negative")
    if config.preview_count < 0:
        raise ValueError("preview_count cannot be negative")
    if config.jpeg_quality < 1 or config.jpeg_quality > 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    if config.progress_update_seconds < 0.0:
        raise ValueError("progress_update_seconds cannot be negative")


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _preflight_sources(backgrounds: list[BackgroundSource], pdfs: list[PdfSource],
                       real_sources: list[RealPieceSource],
                       progress: _ProgressReporter | None = None) -> dict[Path, int]:
    duplicate_backgrounds = _duplicates([source.sha256 for source in backgrounds])
    if duplicate_backgrounds:
        raise ValueError(
            "Duplicate background file content detected; remove duplicate copies: "
            f"{duplicate_backgrounds[:5]}")
    duplicate_pdf_content = _duplicates([source.sha256 for source in pdfs])
    if duplicate_pdf_content:
        raise ValueError(
            "Duplicate PDF file content detected; remove duplicate copies: "
            f"{duplicate_pdf_content[:5]}")
    duplicate_document_ids = _duplicates([source.document_id for source in pdfs])
    if duplicate_document_ids:
        raise ValueError(
            "PDF document_id values must be unique: "
            f"{duplicate_document_ids[:5]}")
    duplicate_real_pairs = _duplicates([
        f"{source.image_sha256}:{source.mask_sha256}" for source in real_sources
    ])
    if duplicate_real_pairs:
        raise ValueError(
            "Duplicate real image/mask content detected in pieces.csv: "
            f"{duplicate_real_pairs[:5]}")

    if progress is not None:
        progress.phase(
            "Source preflight", len(backgrounds) + len(real_sources) + len(pdfs))
    for background in backgrounds:
        try:
            with Image.open(background.path) as handle:
                ImageOps.exif_transpose(handle).convert("RGB").load()
        except Exception as error:
            raise ValueError(
                f"Unreadable background image: {background.path}") from error
        if progress is not None:
            progress.advance(f"background {background.path.name}")

    for source in real_sources:
        try:
            with Image.open(source.image_path) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
                image.load()
            with Image.open(source.mask_path) as handle:
                mask_image = ImageOps.exif_transpose(handle).convert("L")
                mask_image.load()
        except Exception as error:
            raise ValueError(
                f"Unreadable real image/mask pair for {source.piece_id}") from error
        if image.size != mask_image.size:
            raise ValueError(
                f"Real image and mask dimensions differ for {source.piece_id}")
        mask = np.asarray(mask_image, dtype=np.uint8) >= 128
        components = ndimage.label(mask, np.ones((3, 3), dtype=np.uint8))[1]
        if int(mask.sum()) < 50 or bool(mask.all()) or components != 1:
            raise ValueError(
                f"Real mask must contain one connected, non-full paper region: "
                f"{source.mask_path}")
        if progress is not None:
            progress.advance(f"real mask {source.mask_path.name}")

    page_counts: dict[Path, int] = {}
    for pdf in pdfs:
        count = _pdf_page_count(pdf)
        if count < 1:
            raise ValueError(f"PDF contains no pages: {pdf.path}")
        page_counts[pdf.path] = count
        if progress is not None:
            progress.advance(f"PDF {pdf.path.name}")
    return page_counts


def _generate_raster_safe_partition(
        width: int, height: int, target_pieces: int, selected_family: str,
        initial_rng: np.random.Generator, base_seed: int, maximum_attempts: int,
        context: str = "partition",
        ) -> tuple[Partition, np.ndarray, dict[str, Any]]:
    """Generate and rasterize as one deterministic, retryable operation."""
    if maximum_attempts < 1:
        raise ValueError("maximum_attempts must be positive")
    rejections: Counter[str] = Counter()
    fallback_start = max(1, maximum_attempts // 2)
    for attempt in range(maximum_attempts):
        use_fallback = attempt >= fallback_start and selected_family != "sequential"
        trial_family = "sequential" if use_fallback else selected_family
        trial_rng = initial_rng if attempt == 0 else np.random.default_rng(
            _seed(base_seed, "partition_retry", attempt, trial_family))
        try:
            partition = generate_partition(
                width, height, target_pieces, trial_rng, trial_family)
            labels = rasterize_partition(partition, width, height)
        except GeometryError as error:
            rejections[f"{trial_family}: {error}"] += 1
            continue
        return partition, labels, {
            "requested_family": selected_family,
            "accepted_family": partition.family,
            "compiler_attempt": attempt + 1,
            "used_sequential_fallback": use_fallback,
            "compiler_rejections": dict(rejections),
        }
    summary = "; ".join(
        f"{reason} ({count})" for reason, count in sorted(rejections.items()))
    raise GeometryError(
        f"Could not create a raster-safe {target_pieces}-piece partition for "
        f"{context} after {maximum_attempts} deterministic attempts. "
        f"Rejections: {summary or 'unknown'}")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = [
        "sample_id", "split", "sample_kind", "capture_profile", "group_id",
        "document_id", "page_id", "piece_id", "background_asset_id",
        "image_path", "mask_path", "seed", "foreground_fraction",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    os.replace(temporary, path)


def _save_pair(output: Path, split: str, sample_id: str,
               image: np.ndarray, mask: np.ndarray, jpeg_quality: int
               ) -> tuple[str, str]:
    if image.shape[:2] != mask.shape or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Invalid image/mask pair for {sample_id}")
    image_path = output / split / "images" / f"{sample_id}.jpg"
    mask_path = output / split / "masks" / f"{sample_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    image_temp = image_path.with_suffix(".jpg.tmp")
    mask_temp = mask_path.with_suffix(".png.tmp")
    Image.fromarray(image, mode="RGB").save(
        image_temp, format="JPEG", quality=jpeg_quality, subsampling=0,
        optimize=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
        mask_temp, format="PNG", optimize=True)
    os.replace(image_temp, image_path)
    os.replace(mask_temp, mask_path)
    return image_path.relative_to(output).as_posix(), mask_path.relative_to(output).as_posix()


def _normalise_real_pair(source: RealPieceSource, size: int
                         ) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(source.image_path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
    with Image.open(source.mask_path) as handle:
        mask = ImageOps.exif_transpose(handle).convert("L")
    if image.size != mask.size:
        raise ValueError(f"Real image and mask dimensions differ for {source.piece_id}")
    scale = min(size / image.width, size / image.height)
    resized_size = (max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)))
    image = image.resize(resized_size, Image.Resampling.LANCZOS)
    mask = mask.resize(resized_size, Image.Resampling.NEAREST)
    pixels = np.asarray(image, dtype=np.uint8)
    corners = np.concatenate((pixels[:4, :4].reshape(-1, 3),
                              pixels[:4, -4:].reshape(-1, 3),
                              pixels[-4:, :4].reshape(-1, 3),
                              pixels[-4:, -4:].reshape(-1, 3)), axis=0)
    fill = tuple(int(value) for value in np.median(corners, axis=0))
    canvas = Image.new("RGB", (size, size), fill)
    target = Image.new("L", (size, size), 0)
    left, top = (size - image.width) // 2, (size - image.height) // 2
    canvas.paste(image, (left, top))
    target.paste(mask, (left, top))
    result_mask = np.asarray(target, dtype=np.uint8) >= 128
    if int(result_mask.sum()) < 50 or bool(result_mask.all()):
        raise ValueError(f"Real mask became empty/full for {source.piece_id}")
    return np.asarray(canvas, dtype=np.uint8), result_mask


def _background_choice(backgrounds: list[BackgroundSource],
                       rng: np.random.Generator
                       ) -> tuple[str, BackgroundSource | None]:
    if not backgrounds:
        profile = "phone" if rng.random() < 0.5 else "scanner"
        return profile, None
    available = sorted({source.capture_type for source in backgrounds})
    profile = str(rng.choice(available))
    candidates = [source for source in backgrounds if source.capture_type == profile]
    return profile, candidates[int(rng.integers(0, len(candidates)))]


def _render_pdf_page(source: PdfSource, page_index: int, dpi: int) -> np.ndarray:
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
    except ImportError as error:
        raise RuntimeError(
            "PDF generation requires PyMuPDF; install the project's pdf extra"
        ) from error
    try:
        with fitz.open(source.path) as document:
            if page_index < 0 or page_index >= len(document):
                raise ValueError(f"Invalid page {page_index} for {source.path}")
            scale = dpi / 72.0
            pixmap = document[page_index].get_pixmap(
                matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
            return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, 3).copy()
    except Exception as error:
        raise RuntimeError(
            f"Could not render page {page_index + 1} of PDF {source.path}") from error


def _pdf_page_count(source: PdfSource) -> int:
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
    except ImportError as error:
        raise RuntimeError("PDF generation requires PyMuPDF") from error
    try:
        with fitz.open(source.path) as document:
            return len(document)
    except Exception as error:
        raise RuntimeError(f"Could not open PDF {source.path}") from error


def _partition_metadata(partition: Partition, document_id: str, page_index: int,
                        puzzle_id: str, width: int, height: int,
                        compiler_diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "puzzle_id": puzzle_id,
        "document_id": document_id,
        "page_index": page_index,
        "page_size": [width, height],
        "family": partition.family,
        "cut_styles": partition.cut_styles,
        "attempts": partition.attempts,
        "rejections": partition.rejections,
        "compiler": compiler_diagnostics,
        "cut_paths": [
            [[round(float(x), 4), round(float(y), 4)] for x, y in path.coords]
            for path in partition.paths
        ],
        "piece_polygons": [
            [[round(float(x), 4), round(float(y), 4)]
             for x, y in polygon.exterior.coords]
            for polygon in partition.pieces
        ],
    }


def _inventory_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _record(sample_id: str, split: str, sample_kind: str,
            profile: str, group_id: str, document_id: str, page_id: str,
            piece_id: str, background: BackgroundSource | None,
            image_path: str, mask_path: str, seed: int,
            foreground_fraction: float, lineage: dict[str, Any],
            effects: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "sample_id": sample_id,
        "split": split,
        "sample_kind": sample_kind,
        "capture_profile": profile,
        "group_id": group_id,
        "document_id": document_id,
        "page_id": page_id,
        "piece_id": piece_id,
        "background_asset_id": "" if background is None else background.asset_id,
        "background_session_id": "" if background is None else background.session_id,
        "image_path": image_path,
        "mask_path": mask_path,
        "seed": int(seed),
        "foreground_fraction": float(foreground_fraction),
        "lineage": lineage,
        "effects": effects,
    }


def _contact_sheet(output: Path, records: list[dict[str, Any]], count: int) -> None:
    positives = [row for row in records if row["sample_kind"] != "background_only"]
    negatives = [row for row in records if row["sample_kind"] == "background_only"]
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in positives:
        by_kind.setdefault(row["sample_kind"], []).append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < count and any(by_kind.values()):
        for kind in sorted(by_kind):
            if by_kind[kind] and len(selected) < count:
                selected.append(by_kind[kind].pop(0))
    if negatives and len(selected) < count:
        selected.extend(negatives[:count - len(selected)])
    tiles: list[Image.Image] = []
    for row in selected:
        with Image.open(output / row["image_path"]) as handle:
            image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        with Image.open(output / row["mask_path"]) as handle:
            mask = np.asarray(handle.convert("L"), dtype=np.uint8) >= 128
        overlay = image.copy()
        if mask.any():
            boundary = mask ^ ndimage.binary_erosion(mask)
            overlay[boundary] = (30, 255, 70)
        tile = Image.fromarray(overlay).resize((280, 280), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (280, 316), "white")
        canvas.paste(tile, (0, 36))
        ImageDraw.Draw(canvas).text(
            (7, 7), f"{row['sample_kind']} | {row['capture_profile']}", fill="black")
        tiles.append(canvas)
    if not tiles:
        return
    columns = 4
    sheet = Image.new("RGB", (columns * 280,
                              ((len(tiles) + columns - 1) // columns) * 316),
                      "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 280, (index // columns) * 316))
    preview = output / "previews" / "contact_sheet.jpg"
    preview.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(preview, quality=94, subsampling=0)


def validate_generated_dataset(output: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    identifiers = [row["sample_id"] for row in records]
    duplicates = [name for name, count in Counter(identifiers).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate sample IDs: {duplicates[:5]}")
    for row in records:
        image_path, mask_path = output / row["image_path"], output / row["mask_path"]
        if not image_path.is_file() or not mask_path.is_file():
            errors.append(f"{row['sample_id']}: missing image or mask")
            continue
        with Image.open(image_path) as handle:
            image_size = handle.size
        with Image.open(mask_path) as handle:
            mask = np.asarray(handle.convert("L"), dtype=np.uint8)
        if image_size != (mask.shape[1], mask.shape[0]):
            errors.append(f"{row['sample_id']}: image/mask size mismatch")
        unique = set(np.unique(mask).tolist())
        if not unique.issubset({0, 255}):
            errors.append(f"{row['sample_id']}: mask is not binary")
        foreground = mask >= 128
        if row["sample_kind"] == "background_only":
            if foreground.any():
                errors.append(f"{row['sample_id']}: negative mask is not empty")
        else:
            components = ndimage.label(foreground, np.ones((3, 3), dtype=np.uint8))[1]
            if not foreground.any() or foreground.all() or components != 1:
                errors.append(
                    f"{row['sample_id']}: positive mask is empty/full/disconnected")
        if row["sample_kind"] != "real_capture" and row["split"] != "train":
            errors.append(f"{row['sample_id']}: generated sample leaked outside train")
    real_rows = [row for row in records if row["sample_kind"] == "real_capture"]
    real_counts = Counter(row["split"] for row in real_rows)
    real_profiles_by_split = {
        split: dict(Counter(
            row["capture_profile"] for row in real_rows if row["split"] == split))
        for split in ("train", "validation", "calibration", "holdout")
    }
    if len(real_rows) >= 4:
        for split in ("train", "validation", "calibration", "holdout"):
            if real_counts[split] == 0:
                errors.append(
                    f"required split {split!r} has no untouched real captures")
    return {
        "valid": not errors,
        "sample_count": len(records),
        "samples_by_split": dict(Counter(row["split"] for row in records)),
        "samples_by_kind": dict(Counter(row["sample_kind"] for row in records)),
        "samples_by_profile": dict(Counter(row["capture_profile"] for row in records)),
        "real_captures_by_split": dict(real_counts),
        "real_profiles_by_split": real_profiles_by_split,
        "errors": errors,
    }


def generate_dataset(project_root: Path, config: GeneratorConfig | None = None
                     ) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    config = config or GeneratorConfig()
    _validate_config(config)
    progress = _ProgressReporter(
        config.show_progress, config.progress_update_seconds)
    progress.note("Discovering generator inputs...")
    input_root = (project_root / config.input_root).resolve()
    output = (project_root / config.output_root).resolve()
    if input_root == output or input_root in output.parents or output in input_root.parents:
        raise ValueError("input_root and output_root must be separate folder trees")
    source_folders = create_source_layout(input_root)
    backgrounds = discover_backgrounds(source_folders["backgrounds"])
    pdfs = discover_pdfs(source_folders["pdfs"])
    real_sources = discover_real_pieces(source_folders["real_pieces"])
    progress.note(
        f"Found {len(backgrounds):,} backgrounds, {len(pdfs):,} PDFs, "
        f"and {len(real_sources):,} real captures.")

    if not pdfs and not real_sources:
        return {
            "waiting_for_sources": True,
            "input_root": str(input_root),
            "message": "Add PDFs and/or completed real image-mask rows, then run again.",
        }
    if config.require_real_backgrounds and not backgrounds:
        raise RuntimeError(
            f"No clean background images found under {source_folders['backgrounds']}"
        )
    if output.exists() and any(output.rglob("*")):
        raise RuntimeError(
            f"Output folder is not empty: {output}. Rename it before rebuilding."
        )
    pdf_page_counts = _preflight_sources(
        backgrounds, pdfs, real_sources, progress)
    for split_name in ("train", "validation", "calibration", "holdout"):
        (output / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output / split_name / "masks").mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    page_references: list[tuple[PdfSource, int]] = []

    # Untouched real captures are the only samples allowed outside training.
    progress.phase("Real captures", len(real_sources))
    for index, source in enumerate(real_sources):
        image, mask = _normalise_real_pair(source, config.image_size)
        sample_id = f"real_{stable_id(source.image_sha256, source.mask_sha256)}"
        image_path, mask_path = _save_pair(
            output, source.split, sample_id, image, mask, config.jpeg_quality)
        records.append(_record(
            sample_id, source.split, "real_capture", source.capture_type,
            f"real_piece:{source.document_id}:{source.page_id}:{source.piece_id}",
            source.document_id, source.page_id, source.piece_id, None,
            image_path, mask_path, _seed(config.random_seed, "real", index),
            float(mask.mean()),
            {"source_image_sha256": source.image_sha256,
             "source_mask_sha256": source.mask_sha256,
             "capture_session_id": source.capture_session_id},
            {"normalisation": "aspect_fit_with_corner_median_padding"},
        ))
        progress.advance(source.piece_id)

    # Build procedural fragments from PDFs using the new mixed geometry engine.
    total_puzzles = sum(
        min(page_count, config.max_pages_per_pdf)
        if config.max_pages_per_pdf is not None else page_count
        for page_count in pdf_page_counts.values()
    ) * config.puzzles_per_pdf_page
    progress.phase("PDF puzzles", total_puzzles)
    for pdf in pdfs:
        page_count = pdf_page_counts[pdf.path]
        if config.max_pages_per_pdf is not None:
            page_count = min(page_count, config.max_pages_per_pdf)
        for page_index in range(page_count):
            page_references.append((pdf, page_index))
            page = _render_pdf_page(pdf, page_index, config.pdf_dpi)
            height, width = page.shape[:2]
            for puzzle_index in range(config.puzzles_per_pdf_page):
                geometry_seed = _seed(config.random_seed, pdf.sha256, page_index,
                                      puzzle_index, "geometry")
                geometry_rng = np.random.default_rng(geometry_seed)
                target = int(geometry_rng.integers(
                    config.min_pieces_per_puzzle,
                    config.max_pieces_per_puzzle + 1))
                # Crossing topology requires a composite count; auto selection
                # handles this by retrying through a compatible family.
                family = str(geometry_rng.choice(
                    ("sequential", "global_freeform", "crossing", "parallel"),
                    p=(0.60, 0.20, 0.10, 0.10)))
                if family == "crossing" and all(target % factor for factor in range(2, target)):
                    family = "sequential"
                puzzle_id = f"puz_{stable_id(pdf.sha256, page_index, puzzle_index)}"
                context = (
                    f"{pdf.path.name}, page {page_index + 1}, "
                    f"puzzle {puzzle_index + 1}")
                partition: Partition | None = None
                compiler_diagnostics: dict[str, Any] | None = None
                prepared: list[dict[str, Any]] = []
                puzzle_rejections: Counter[str] = Counter()
                for puzzle_attempt in range(config.puzzle_render_attempts):
                    trial_geometry_seed = (
                        geometry_seed if puzzle_attempt == 0 else
                        _seed(geometry_seed, "puzzle_render_retry", puzzle_attempt)
                    )
                    trial_rng = (
                        geometry_rng if puzzle_attempt == 0 else
                        np.random.default_rng(trial_geometry_seed)
                    )
                    try:
                        candidate_partition, labels, diagnostics = (
                            _generate_raster_safe_partition(
                                width, height, target, family, trial_rng,
                                trial_geometry_seed,
                                config.geometry_raster_attempts,
                                context=context,
                            )
                        )
                        candidate_prepared: list[dict[str, Any]] = []
                        for piece_index in range(target):
                            piece = piece_from_page(
                                page, labels == piece_index + 1)
                            for variant in range(
                                    config.variants_per_synthetic_piece):
                                seed = _seed(
                                    config.random_seed, puzzle_id, piece_index,
                                    variant, "capture")
                                rng = np.random.default_rng(seed)
                                profile, background = _background_choice(
                                    backgrounds, rng)
                                image, mask, effects = render_capture(
                                    piece,
                                    None if background is None else background.path,
                                    profile, config.image_size, seed,
                                    maximum_attempts=config.capture_pose_attempts)
                                candidate_prepared.append({
                                    "piece_index": piece_index,
                                    "variant": variant,
                                    "seed": seed,
                                    "profile": profile,
                                    "background": background,
                                    "image": image,
                                    "mask": mask,
                                    "effects": effects,
                                })
                    except (CaptureRenderError, GeometryError) as error:
                        reason = f"{type(error).__name__}: {error}"
                        puzzle_rejections[reason] += 1
                        progress.note(
                            f"Rejected {context} attempt {puzzle_attempt + 1}/"
                            f"{config.puzzle_render_attempts}; resampling the "
                            f"whole puzzle ({error}).")
                        continue
                    partition = candidate_partition
                    compiler_diagnostics = diagnostics
                    compiler_diagnostics["puzzle_render_attempt"] = puzzle_attempt + 1
                    compiler_diagnostics["puzzle_rejections"] = dict(
                        puzzle_rejections)
                    prepared = candidate_prepared
                    break
                if partition is None or compiler_diagnostics is None:
                    summary = "; ".join(
                        f"{reason} ({count})"
                        for reason, count in sorted(puzzle_rejections.items()))
                    raise CaptureRenderError(
                        f"Could not create a transform-safe puzzle for {context} "
                        f"after {config.puzzle_render_attempts} complete puzzle "
                        f"attempts. Rejections: {summary or 'unknown'}")
                _atomic_json(
                    output / "geometry" / f"{puzzle_id}.json",
                    _partition_metadata(partition, pdf.document_id, page_index,
                                        puzzle_id, width, height,
                                        compiler_diagnostics))
                for sample in prepared:
                    piece_index = int(sample["piece_index"])
                    variant = int(sample["variant"])
                    seed = int(sample["seed"])
                    profile = str(sample["profile"])
                    background = sample["background"]
                    image = sample["image"]
                    mask = sample["mask"]
                    effects = sample["effects"]
                    sample_id = (
                        f"syn_{stable_id(puzzle_id, piece_index, variant, seed)}")
                    image_path, mask_path = _save_pair(
                        output, "train", sample_id, image, mask,
                        config.jpeg_quality)
                    records.append(_record(
                        sample_id, "train", "procedural_piece", profile,
                        f"synthetic_page:{pdf.document_id}:{page_index}",
                        pdf.document_id, str(page_index),
                        f"{puzzle_id}_piece_{piece_index:03d}", background,
                        image_path, mask_path, seed, float(mask.mean()),
                        {"pdf_sha256": pdf.sha256,
                         "pdf_license": pdf.license_name,
                         "puzzle_id": puzzle_id,
                         "requested_geometry_family": family,
                         "geometry_family": partition.family,
                         "geometry_compiler_attempt":
                             compiler_diagnostics["compiler_attempt"],
                         "puzzle_render_attempt":
                             compiler_diagnostics["puzzle_render_attempt"],
                         "cut_styles": partition.cut_styles},
                        effects,
                    ))
                progress.advance(
                    f"{pdf.path.name}, page {page_index + 1}, "
                    f"puzzle {puzzle_index + 1} ({target} pieces)")

    # Use one clean canonical capture per training piece for derived examples.
    grouped: dict[tuple[str, str, str], list[RealPieceSource]] = {}
    for source in real_sources:
        if source.split == "train":
            key = (source.document_id, source.page_id, source.piece_id)
            grouped.setdefault(key, []).append(source)
    canonical_sources: list[RealPieceSource] = []
    for _, group in sorted(grouped.items()):
        explicit = [source for source in group if source.canonical_for_compositing]
        canonical_sources.append(explicit[0] if explicit else sorted(
            group, key=lambda source: source.image_sha256)[0])

    derived_per_piece = config.real_composites_per_piece
    if page_references:
        derived_per_piece += config.real_retextures_per_piece
    progress.phase(
        "Real-piece augmentations", len(canonical_sources) * derived_per_piece)
    for source in canonical_sources:
        piece = load_real_piece(source.image_path, source.mask_path)
        group_id = (
            f"real_piece:{source.document_id}:{source.page_id}:{source.piece_id}")
        for variant in range(config.real_composites_per_piece):
            seed = _seed(config.random_seed, source.document_id, source.page_id,
                         source.piece_id, variant, "real_composite")
            rng = np.random.default_rng(seed)
            profile, background = _background_choice(backgrounds, rng)
            image, mask, effects = render_capture(
                piece, None if background is None else background.path,
                profile, config.image_size, seed,
                maximum_attempts=config.capture_pose_attempts)
            sample_id = f"realcomp_{stable_id(source.image_sha256, variant, seed)}"
            image_path, mask_path = _save_pair(
                output, "train", sample_id, image, mask, config.jpeg_quality)
            records.append(_record(
                sample_id, "train", "real_piece_composite", profile, group_id,
                source.document_id, source.page_id, source.piece_id, background,
                image_path, mask_path, seed, float(mask.mean()),
                {"source_image_sha256": source.image_sha256,
                 "source_mask_sha256": source.mask_sha256,
                 "capture_session_id": source.capture_session_id},
                effects,
            ))
            progress.advance(f"composite {source.piece_id}")

        if page_references:
            for variant in range(config.real_retextures_per_piece):
                seed = _seed(config.random_seed, source.document_id, source.page_id,
                             source.piece_id, variant, "retexture")
                rng = np.random.default_rng(seed)
                pdf, page_index = page_references[int(rng.integers(0, len(page_references)))]
                page = _render_pdf_page(pdf, page_index, config.pdf_dpi)
                retextured = retexture_real_shape(piece, page, rng)
                profile, background = _background_choice(backgrounds, rng)
                image, mask, effects = render_capture(
                    retextured, None if background is None else background.path,
                    profile, config.image_size, seed,
                    maximum_attempts=config.capture_pose_attempts)
                sample_id = f"retex_{stable_id(source.mask_sha256, pdf.sha256, page_index, variant)}"
                image_path, mask_path = _save_pair(
                    output, "train", sample_id, image, mask, config.jpeg_quality)
                records.append(_record(
                    sample_id, "train", "real_shape_retextured", profile,
                    group_id, source.document_id, source.page_id,
                    source.piece_id, background, image_path, mask_path, seed,
                    float(mask.mean()),
                    {"source_mask_sha256": source.mask_sha256,
                     "texture_pdf_sha256": pdf.sha256,
                     "texture_page_index": page_index,
                     "texture_document_id": pdf.document_id},
                    effects,
                ))
                progress.advance(f"retexture {source.piece_id}")

    positive_train_count = sum(
        row["split"] == "train" and row["sample_kind"] != "background_only"
        for row in records)
    negative_count = (
        0
        if config.background_negative_ratio <= 0.0
        else max(1, round(positive_train_count * config.background_negative_ratio))
    )
    progress.phase("Background-only samples", negative_count)
    for index in range(negative_count):
        seed = _seed(config.random_seed, "background_only", index)
        rng = np.random.default_rng(seed)
        profile, background = _background_choice(backgrounds, rng)
        image, mask, effects = render_background_only(
            None if background is None else background.path,
            profile, config.image_size, seed)
        sample_id = f"negative_{stable_id(seed, index)}"
        image_path, mask_path = _save_pair(
            output, "train", sample_id, image, mask, config.jpeg_quality)
        records.append(_record(
            sample_id, "train", "background_only", profile,
            f"background:{'' if background is None else background.session_id}",
            "negative", "negative", "negative", background,
            image_path, mask_path, seed, 0.0,
            {"background_sha256": "" if background is None else background.sha256},
            effects,
        ))
        progress.advance(f"negative {index + 1}")

    progress.note("Validating generated images, masks, and split policy...")
    records.sort(key=lambda row: (row["split"], row["sample_id"]))
    validation = validate_generated_dataset(output, records)
    _atomic_jsonl(output / "samples.jsonl", records)
    _atomic_jsonl(output / "recipes.jsonl", records)
    _atomic_csv(output / "samples.csv", records)
    inventory = {
        "backgrounds": [
            {**asdict(source), "path": _inventory_path(source.path, project_root)}
            for source in backgrounds],
        "pdfs": [
            {**asdict(source), "path": _inventory_path(source.path, project_root)}
            for source in pdfs],
        "real_pieces": [
            {**asdict(source),
             "image_path": _inventory_path(source.image_path, project_root),
             "mask_path": _inventory_path(source.mask_path, project_root)}
            for source in real_sources],
    }
    _atomic_json(output / "source_inventory.json", inventory)
    report = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "config": {**asdict(config),
                   "input_root": str(config.input_root),
                   "output_root": str(config.output_root)},
        "source_counts": {
            "backgrounds": len(backgrounds),
            "pdfs": len(pdfs),
            "real_captures": len(real_sources),
            "canonical_real_pieces": len(canonical_sources),
        },
        **validation,
        "split_policy": (
            "Generated derivatives are train-only. Validation, calibration, and "
            "holdout contain only real captures grouped by physical piece "
            "in pieces.csv."
        ),
    }
    _atomic_json(output / "dataset_report.json", report)
    _contact_sheet(output, records, config.preview_count)
    if not validation["valid"]:
        raise RuntimeError(
            f"Generated dataset failed validation: {validation['errors'][:8]}")
    progress.finish(len(records))
    return report
