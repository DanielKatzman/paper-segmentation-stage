"""Resumable annotation session driven by the capture manifest."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from universal_segmenter.annotation.editor import MaskEditor, create_overlay
from universal_segmenter.annotation.split_policy import (
    SPLIT_LABELS,
    set_group_split,
)
from universal_segmenter.data.manifest import (
    capture_output_relative_path,
    read_jsonl,
    write_jsonl,
)
from universal_segmenter.inference.preprocess import load_rgb


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    mode = "L" if array.ndim == 2 else "RGB"
    Image.fromarray(array, mode=mode).save(temporary, format="PNG", compress_level=1)
    os.replace(temporary, path)


def _review_overlay(rgb: np.ndarray, mask: np.ndarray,
                    maximum_side: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = min(1.0, maximum_side / max(height, width))
    if scale >= 1.0:
        return create_overlay(rgb, mask)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small_rgb = np.asarray(Image.fromarray(rgb, mode="RGB").resize(
        size, Image.Resampling.LANCZOS), dtype=np.uint8)
    small_mask = np.asarray(Image.fromarray(
        mask.astype(np.uint8) * 255, mode="L").resize(
            size, Image.Resampling.NEAREST), dtype=np.uint8) >= 128
    return create_overlay(small_rgb, small_mask)


def _preview_sheet(records: list[dict[str, Any]], project: Path, output: Path) -> None:
    tiles = []
    for record in records:
        overlay_value = str(record.get("overlay_path", "")).strip()
        if not overlay_value:
            continue
        path = project / overlay_value
        if not path.is_file():
            continue
        with Image.open(path) as source:
            tile = source.convert("RGB")
            tile.thumbnail((360, 320), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (380, 370), (238, 238, 238))
        canvas.paste(tile, ((380 - tile.width) // 2, 42))
        ImageDraw.Draw(canvas).text(
            (8, 12),
            f"{record['capture_id']} | {record.get('capture_type', '?')} | "
            f"{SPLIT_LABELS.get(str(record.get('dataset_split', '')), '?')}",
            fill=(10, 10, 10))
        tiles.append(canvas)
    if not tiles:
        return
    columns = min(3, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 380, rows * 370), (220, 220, 220))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 380, (index // columns) * 370))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=93, subsampling=0)


def reopen_missing_annotation_outputs(
        project: Path, records: list[dict[str, Any]], output: Path,
        ) -> list[dict[str, Any]]:
    """Return reviewed records with a missing mask/overlay to the work queue.

    Deleting either generated file is an intentional, filesystem-level signal
    that the capture should be corrected again. The surviving file is kept and
    will be overwritten together with its missing partner on the next save.
    """
    reopened: list[dict[str, Any]] = []
    for record in records:
        if record.get("annotation_status") != "reviewed":
            continue
        relative_output = capture_output_relative_path(record)
        explicit_mask = str(record.get("mask_path", "")).strip()
        explicit_overlay = str(record.get("overlay_path", "")).strip()
        mask_path = (
            project / explicit_mask
            if explicit_mask else output / "masks" / relative_output
        )
        overlay_path = (
            project / explicit_overlay
            if explicit_overlay
            else output / "overlays" / relative_output.with_suffix(".jpg")
        )
        missing = [
            name for name, path in (("mask", mask_path), ("overlay", overlay_path))
            if not path.is_file()
        ]
        if not missing:
            record.pop("reannotation_reason", None)
            record.pop("missing_annotation_outputs", None)
            continue
        record["annotation_status"] = "unlabeled"
        record["reannotation_reason"] = "missing_generated_output"
        record["missing_annotation_outputs"] = missing
        reopened.append(record)
    return reopened


def annotate_manifest(project: Path, manifest_path: Path,
                      output_folder: Path = Path("data/annotations"),
                      review_saved: bool = False,
                      pdf_dpi: int = 300,
                      materialize_annotation_images: bool = False,
                      overlay_max_side: int = 1600,
                      preview_interval: int = 20,
                      auto_regularize: bool = True,
                      regularize_closing_iterations: int = 1,
                      proposal_provider: Any | None = None,
                      ) -> dict[str, int | bool]:
    project = Path(project)
    manifest_path = project / manifest_path
    records = read_jsonl(manifest_path)
    if not records:
        raise RuntimeError(f"capture manifest is empty: {manifest_path}")
    if proposal_provider is None:
        raise ValueError("annotation requires the calibrated SegFormer provider")
    output = project / output_folder
    reopened = reopen_missing_annotation_outputs(project, records, output)
    if reopened:
        write_jsonl(manifest_path, records)
        print(
            f"Reopened {len(reopened)} reviewed capture(s) with missing "
            "mask/overlay files:",
            flush=True,
        )
        for record in reopened:
            missing = ", ".join(record["missing_annotation_outputs"])
            print(f"  - {record['capture_id']}: missing {missing}", flush=True)
    saved_this_run = 0
    preview_dirty = False

    for index, record in enumerate(records, start=1):
        if record.get("annotation_status") == "rejected":
            continue
        if record.get("annotation_status") == "reviewed" and not review_saved:
            continue
        record.setdefault("physical_piece_id", str(
            record.get("fragment_id") or record["capture_id"]))
        record.setdefault("split_group_id", str(record["capture_id"]))
        record.setdefault("dataset_split", "train")
        record.setdefault("split_assignment_source", "automatic")
        source_path = project / record["image_path"]
        if not source_path.is_file():
            materialized = str(record.get("annotation_image_path", "")).strip()
            candidate = project / materialized if materialized else source_path
            if candidate.is_file():
                source_path = candidate
        rgb = load_rgb(source_path, pdf_dpi=pdf_dpi)
        proposal_sources: dict[str, str] = {}
        proposal_key = str(proposal_provider.primary_model)
        proposal_choices = list(proposal_provider.choices)
        proposal_name = dict(proposal_choices)[proposal_key]

        def proposal_loader(selected_key: str) -> np.ndarray:
            value, source = proposal_provider.get(record, rgb, selected_key)
            proposal_sources[selected_key] = source
            return value

        proposal = proposal_loader(proposal_key)
        background_status = proposal_provider.status_text
        relative_output = capture_output_relative_path(record)
        explicit_mask = str(record.get("mask_path", "")).strip()
        saved_mask_path = project / explicit_mask if explicit_mask else None
        mask_path = (
            saved_mask_path if saved_mask_path is not None and saved_mask_path.is_file()
            else output / "masks" / relative_output
        )
        if mask_path.is_file():
            with Image.open(mask_path) as source:
                current = np.asarray(source.convert("L"), dtype=np.uint8) >= 128
            if current.shape != proposal.shape:
                raise ValueError(f"saved mask shape mismatch: {mask_path}")
        else:
            current = proposal.copy()
        editor = MaskEditor(
            rgb, current, proposal,
            f"Universal paper mask {index}/{len(records)} - {record['capture_id']}",
            auto_regularize=auto_regularize,
            regularize_closing_iterations=regularize_closing_iterations,
            proposal_key=proposal_key,
            proposal_name=proposal_name,
            proposal_choices=proposal_choices,
            proposal_loader=proposal_loader,
            background_status=background_status,
            dataset_split=str(record.get("dataset_split", "train")),
            dataset_split_choices=list(SPLIT_LABELS.items()),
        )
        action, corrected = editor.run()
        if action == "quit":
            break
        if not corrected.any() or corrected.all():
            raise RuntimeError(f"refusing empty/full mask for {record['capture_id']}")

        image_path = output / "images" / relative_output
        explicit_overlay = str(record.get("overlay_path", "")).strip()
        overlay_path = (
            project / explicit_overlay
            if explicit_overlay
            else output / "overlays" / relative_output.with_suffix(".jpg")
        )
        if materialize_annotation_images or source_path.suffix.lower() == ".pdf":
            _save_png(image_path, rgb)
            annotation_image_path = image_path.relative_to(project).as_posix()
        else:
            annotation_image_path = str(record["image_path"])
        _save_png(mask_path, corrected.astype(np.uint8) * 255)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = overlay_path.with_suffix(".jpg.tmp")
        Image.fromarray(_review_overlay(rgb, corrected, overlay_max_side)).save(
            temporary, format="JPEG", quality=94, subsampling=0)
        os.replace(temporary, overlay_path)

        record["annotation_status"] = "reviewed"
        record["annotation_version"] = int(record.get("annotation_version", 0)) + 1
        record["mask_path"] = mask_path.relative_to(project).as_posix()
        record["annotation_image_path"] = annotation_image_path
        record["overlay_path"] = overlay_path.relative_to(project).as_posix()
        selected_key = editor.selected_proposal_key
        record["proposal_model_key"] = selected_key
        record["proposal_source"] = proposal_sources.get(
            selected_key, proposal_sources[proposal_key])
        record["proposal_changed_fraction"] = float(
            np.mean(corrected != editor.proposal))
        record["mask_area_fraction"] = float(corrected.mean())
        initial_split = str(record.get("dataset_split", "train"))
        selected_split = str(getattr(
            editor, "selected_dataset_split", initial_split))
        if selected_split != initial_split:
            set_group_split(
                records, str(record["split_group_id"]), selected_split)
        record.pop("reannotation_reason", None)
        record.pop("missing_annotation_outputs", None)
        write_jsonl(manifest_path, records)
        saved_this_run += 1
        preview_dirty = True
        if preview_interval > 0 and saved_this_run % preview_interval == 0:
            _preview_sheet(records, project, output / "annotation_preview_sheet.jpg")
            preview_dirty = False

    if preview_dirty:
        _preview_sheet(records, project, output / "annotation_preview_sheet.jpg")

    reviewed = sum(record.get("annotation_status") == "reviewed" for record in records)
    return {
        "complete": reviewed == sum(record.get("annotation_status") != "rejected" for record in records),
        "reviewed": reviewed,
        "total": len(records),
        "reopened_missing_outputs": len(reopened),
        "captures_by_split": dict(Counter(
            str(record.get("dataset_split", "unassigned"))
            for record in records)),
    }
