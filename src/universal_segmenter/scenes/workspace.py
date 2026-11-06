"""Shared review operations, capture discovery and collision-free piece export."""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from .core import (Scene, SceneStore, atomic_bytes, bbox, canonical, digest,
                   file_hash, instance_errors, png_bytes, validate_scene, write_json)

PROPOSAL_POLICY = "all-components-native-1.1"


def _dirty(scene: Scene) -> None:
    scene.record.update(state="in_progress", clean_background=False)
    scene.record.pop("empty_confirmed", None)


def add_instance(scene: Scene, mask: np.ndarray, *, accepted: bool = False,
                 origin: str = "manual", instance_id: str | None = None,
                 parents: list[str] | None = None) -> str:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != scene.labels.shape or not mask.any():
        raise ValueError("A piece needs a nonempty native-size mask")
    if np.any(mask & (scene.labels > 0)):
        raise ValueError("Piece overlaps another instance; edit, split or merge explicitly")
    if accepted and instance_errors(mask):
        raise ValueError(f"Piece needs correction: {instance_errors(mask)}")
    label = max((i["label"] for i in scene.record["instances"]), default=0) + 1
    ident = instance_id or "piece_" + uuid.uuid4().hex[:24]
    scene.record["instances"].append({"instance_id": ident, "label": label,
        "state": "accepted" if accepted else "proposed", "bbox_xywh": bbox(mask),
        "origin": origin, "parents": parents or [], "physical_piece_id": ""})
    scene.labels[mask] = label
    _dirty(scene)
    return ident


def instance(scene: Scene, ident: str) -> dict[str, Any]:
    return next(i for i in scene.record["instances"] if i["instance_id"] == ident)


def edit_instance(scene: Scene, ident: str, mask: np.ndarray, *, accept: bool = False) -> None:
    item = instance(scene, ident)
    if item["state"] == "rejected":
        raise ValueError("Restore a revision or add a new instance instead of editing a rejection")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != scene.labels.shape or not mask.any():
        raise ValueError("Correction must be nonempty and native-size")
    if np.any(mask & (scene.labels != 0) & (scene.labels != item["label"])):
        raise ValueError("Correction overlaps another piece")
    if accept and instance_errors(mask):
        raise ValueError(f"Piece needs correction: {instance_errors(mask)}")
    scene.labels[scene.labels == item["label"]] = 0
    scene.labels[mask] = item["label"]
    item.update(bbox_xywh=bbox(mask), state="accepted" if accept else "proposed", origin="manual_review")
    _dirty(scene)


def accept_instance(scene: Scene, ident: str) -> None:
    item = instance(scene, ident)
    edit_instance(scene, ident, scene.labels == item["label"], accept=True)


def reject_instance(scene: Scene, ident: str, reason: str = "false_detection") -> None:
    item = instance(scene, ident)
    scene.labels[scene.labels == item["label"]] = 0
    item.update(state="rejected", rejection_reason=reason)
    _dirty(scene)


def split_instance(scene: Scene, ident: str) -> list[str]:
    item = instance(scene, ident)
    labels, count = ndimage.label(scene.labels == item["label"])
    if count < 2:
        raise ValueError("Erase the incorrect bridge first, then split disconnected regions")
    reject_instance(scene, ident, "split_parent")
    return [add_instance(scene, labels == label, origin="split", parents=[ident])
            for label in range(1, count + 1)]


def merge_instances(scene: Scene, identities: Iterable[str]) -> str:
    identities = list(dict.fromkeys(identities))
    if len(identities) < 2:
        raise ValueError("Select at least two pieces to merge")
    items = [instance(scene, ident) for ident in identities]
    if any(i["state"] == "rejected" for i in items):
        raise ValueError("Cannot merge rejected instances")
    merged = np.isin(scene.labels, [i["label"] for i in items])
    for ident in identities:
        reject_instance(scene, ident, "merge_parent")
    # Disconnected proposal is deliberately editable, but cannot be accepted yet.
    return add_instance(scene, merged, origin="merge", parents=identities)


def finish_scene(scene: Scene, *, empty: bool = False, clean_background: bool = False) -> None:
    if empty:
        if scene.active:
            raise ValueError("Reject/remove all false proposals before confirming an empty capture")
        scene.record.update(state="confirmed_empty", empty_confirmed=True,
                            clean_background=bool(clean_background))
    else:
        if not scene.active or any(i["state"] != "accepted" for i in scene.active):
            raise ValueError("Review every piece and add all missed pieces before completing the scene")
        scene.record.update(state="complete", clean_background=False)
    validate_scene(scene, require_complete=True)


def propose(scene: Scene, probability: np.ndarray, threshold: float,
            model_identity: str, minimum_pixels: int = 4) -> None:
    if scene.record["instances"] or scene.eligible or scene.labels.any():
        raise ValueError("Proposals cannot replace existing annotations; use a new scene revision explicitly")
    p = np.asarray(probability, dtype=np.float32)
    if (p.shape != scene.labels.shape or not np.isfinite(p).all()
            or p.min() < 0 or p.max() > 1 or not 0 < threshold < 1 or minimum_pixels < 1):
        raise ValueError("Invalid probability map, threshold or component minimum")
    labels, count = ndimage.label(p >= threshold, np.ones((3, 3), bool))
    areas = np.bincount(labels.ravel())
    excluded = []
    for label, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None:
            continue
        if areas[label] < minimum_pixels:
            excluded.append({"pixels": int(areas[label]), "bbox_xywh":
                             [region[1].start, region[0].start,
                              region[1].stop - region[1].start, region[0].stop - region[0].start]})
            continue
        # Work inside each component's bounds, rather than re-scanning a phone
        # photo for every piece. Thresholding and native masks are unchanged.
        mask = labels[region] == label
        box = [region[1].start, region[0].start,
               region[1].stop-region[1].start, region[0].stop-region[0].start]
        key = hashlib.sha256(canonical([list(labels.shape), box]) + mask.tobytes()).hexdigest()[:20]
        value = len(scene.record["instances"]) + 1
        scene.labels[region][mask] = value
        scene.record["instances"].append({"instance_id": "piece_" + key, "label": value,
            "state": "proposed", "bbox_xywh": box, "origin": "model_proposal",
            "parents": [], "physical_piece_id": "", "mean_probability": float(p[region][mask].mean())})
    _dirty(scene)
    scene.record["proposal"] = {"policy": PROPOSAL_POLICY, "model_identity": model_identity,
        "threshold": threshold, "minimum_pixels": minimum_pixels,
        "raw_component_count": count, "excluded_specks": excluded}


def discover_into(store: SceneStore, source: Path, metadata: dict[str, Any], dpi: int = 300,
                  *, progress: Callable[[str], None] | None = None,
                  check_cancelled: Callable[[], None] | None = None) -> dict[str, int]:
    source = Path(source).resolve()
    if not source.is_dir() or store.root == source or store.root.is_relative_to(source):
        raise ValueError("Source must exist and workspace must be outside the input folder")
    if not 36 <= dpi <= 600:
        raise ValueError("PDF DPI must be between 36 and 600")
    added = reused = 0
    paths = [p for p in sorted(source.rglob("*")) if p.is_file() and p.suffix.lower() in
             {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}]
    for number, path in enumerate(paths, 1):
        if check_cancelled:
            check_cancelled()
        if progress:
            progress(f"Importing {number}/{len(paths)}: {path.name}")
        base = {**metadata, "source_name": path.relative_to(source).as_posix(),
                "source_sha256": file_hash(path)}
        def save(rgb: np.ndarray, page: int | None) -> None:
            nonlocal added, reused
            if check_cancelled:
                check_cancelled()
            if progress and page:
                progress(f"Importing {number}/{len(paths)}: {path.name}, PDF page {page}")
            if rgb.shape[0] * rgb.shape[1] > 60_000_000:
                raise ValueError("Capture exceeds the 60-million-native-pixel memory budget")
            before = len(store.ids())
            store.add(rgb, {**base, "pdf_page": page, "pdf_dpi": dpi if page else None})
            added += len(store.ids()) > before
            reused += len(store.ids()) == before
        if path.suffix.lower() == ".pdf":
            import pymupdf
            with pymupdf.open(path) as pdf:
                for number, page in enumerate(pdf, 1):
                    if page.rect.width * page.rect.height * (dpi / 72)**2 > 60_000_000:
                        raise ValueError("PDF page exceeds the native-pixel budget at this DPI")
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72),
                                         colorspace=pymupdf.csRGB, alpha=False)
                    save(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy(), number)
        else:
            with Image.open(path) as image:
                if image.width * image.height > 60_000_000:
                    raise ValueError("Capture exceeds the 60-million-native-pixel memory budget")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError(f"Use PDF or separate images for multi-frame input: {path}")
                save(np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8), None)
    return {"added": added, "reused": reused, "total": len(store.ids())}


def publish_pieces(store: SceneStore, output: Path, *, include_generated_fixtures: bool = False) -> dict[str, Any]:
    """Publish approved individual observations; scene-training completeness is separate."""
    import os
    import shutil
    import tempfile
    from universal_segmenter.pipeline.runner import _materialize
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(store.root) or store.root.is_relative_to(output):
        raise ValueError("Publication must be new and separate from the workspace")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, inventory = [], []
    skipped_without_manual_review = 0
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        target = Path(tmp) / "publication"
        target.mkdir()
        for ident in store.ids():
            scene = store.load(ident)
            rgb = store.rgb(scene)
            source_rel = "sources/" + ident + ".png"
            from .core import contained
            (target / source_rel).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(contained(store.root,scene.record["image_path"]),target / source_rel)
            accepted = []
            for item in scene.active:
                if item["state"] != "accepted":
                    continue
                manual = item.get("origin") in {"manual", "manual_review"}
                if not manual and not (include_generated_fixtures and item.get("origin") in
                                      {"synthetic_truth", "selfcheck_fixture"}):
                    skipped_without_manual_review += 1
                    continue
                observation = "capture_" + digest([ident, item["instance_id"]])
                mask = scene.labels == item["label"]
                if instance_errors(mask):
                    raise ValueError("Approved piece failed structural validation")
                paths = _materialize(target, observation, rgb, mask)
                accepted.append(observation)
                rows.append({"schema_version": "paper-segmentation-manifest-1.0",
                    "capture_id": observation, "candidate_id": item["instance_id"],
                    "source_capture_id": ident, "source_path": str(output / source_rel),
                    "source_sha256": scene.record["image_sha256"],
                    "original_size_hw": list(rgb.shape[:2]), "bbox_xywh": bbox(mask),
                    "mask_path": paths["mask"], "piece_path": paths["piece"],
                    "piece_mask_path": paths["piece_mask"], "overlay_path": paths["overlay"],
                    "accepted": True, "review_status": "manually_reviewed" if manual else "auto_accepted", "reason_codes": [],
                    "decision_source": "manual_review" if manual else item["origin"], "model_version":
                    (scene.record.get("proposal") or {}).get("model_identity", "manual"),
                    "annotation_revision": scene.record["revision"]})
            inventory.append({"source_capture_id": ident, "annotation_state": scene.record["state"],
                              "approved_observation_ids": accepted, "metadata": scene.record["metadata"]})
        atomic_bytes(target / "segmentation_manifest.jsonl", b"".join(canonical(row) + b"\n" for row in rows))
        report = {"schema_version": "paper-scene-publication-1.0", "complete":
                  bool(inventory) and not skipped_without_manual_review and all(store.load(i).eligible for i in store.ids()), "source_count": len(inventory),
                  "skipped_without_manual_review": skipped_without_manual_review,
                  "generated_fixture_export": include_generated_fixtures,
                  "approved_piece_count": len(rows), "physical_inventory_confirmed": False}
        write_json(target / "segmentation_run.json", report)
        write_json(target / "capture_inventory.json", inventory)
        write_json(target / "review_publication.json", {"policy": "scene-approved-pieces-1.0",
            "artifacts": {p.relative_to(target).as_posix(): file_hash(p) for p in sorted(target.rglob("*")) if p.is_file()}})
        os.replace(target, output)
    return report


def clone_workspace(source: SceneStore, output: Path) -> dict[str,int]:
    """Start a mutable annotation revision without changing a checked collection."""
    import os
    import shutil
    import tempfile
    from .core import contained
    source.verify()
    output=Path(output).resolve()
    if output.exists() or output.is_relative_to(source.root) or source.root.is_relative_to(output):
        raise ValueError("Clone workspace must be new and separate from its source")
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as folder:
        target=Path(folder)/"workspace"
        target.mkdir()
        for ident in source.ids():
            scene=source.load(ident)
            for key in ("image_path","labels_path","union_path"):
                path=contained(target,scene.record[key])
                path.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(contained(source.root,scene.record[key]),path)
            write_json(target/"scenes"/ident/"current.json",scene.record)
            write_json(target/"scenes"/ident/(scene.record["revision"]+".json"),scene.record)
        write_json(target/"workspace.json",{"cloned_from_checked_collection":
            (source.root/"collection.json").exists(),"source_capture_ids":source.ids()})
        os.replace(target,output)
    return SceneStore(output).verify()
