"""Portable checked collections and non-destructive central import."""
from __future__ import annotations

import os
import copy
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable

from .core import (Scene, SceneStore, atomic_bytes, canonical, contained, digest,
                   exclusive, file_hash, read_json, validate_scene, write_json)

PACKAGE_SCHEMA = "paper-scene-collection-1.0"


def annotation_key(scene: Scene) -> str:
    import hashlib
    masks = sorted(hashlib.sha256((scene.labels == i["label"]).tobytes()).hexdigest()
                   for i in scene.active)
    return digest([scene.record["state"], masks, scene.record.get("clean_background", False),
                   {k: scene.record["metadata"].get(k) for k in
                    ("document_id", "page_id", "capture_session_id", "capture_profile", "split", "split_group_id")},
                   sorted(i.get("physical_piece_id", "") for i in scene.active)])


def export_collection(store: SceneStore, destination: Path, *, include_unfinished: bool = False,
                      progress: Callable[[str], None] | None = None,
                      check_cancelled: Callable[[], None] | None = None) -> dict[str, Any]:
    records = [store.record(i) for i in store.ids()]
    selected_records = [r for r in records if r["state"] in {"complete","confirmed_empty"} or include_unfinished]
    if not selected_records:
        raise ValueError("No complete/confirmed-empty scenes to export")
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(store.root):
        raise ValueError("Choose a new ZIP outside the annotation workspace")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as folder:
        root = Path(folder) / "package"
        root.mkdir()
        copied, identities = [], []
        for number, record in enumerate(selected_records, 1):
            if check_cancelled:
                check_cancelled()
            if progress:
                progress(f"Checking and exporting reviewed image {number}/{len(selected_records)}")
            scene = store.load(record["capture_id"])
            if scene.record["revision"] != record["revision"]:
                raise ValueError("Annotation changed during export; retry after saving")
            # Verify selected artifacts once; don't decode omitted, unreviewed photos.
            store.rgb(scene)
            from PIL import Image
            import numpy as np
            from scipy import ndimage
            union_path = contained(store.root, scene.record["union_path"])
            with Image.open(union_path) as image:
                union = np.asarray(image)
            if (file_hash(union_path) != scene.record["union_sha256"]
                    or not np.array_equal(union,(scene.labels>0).astype(np.uint8)*255)):
                raise ValueError("Union mask does not match instance labels")
            identities.append([scene.record["capture_id"],annotation_key(scene)])
            r = copy.deepcopy(scene.record)
            for asset in r.get("metadata",{}).get("recipe",{}).get("sources",[]):
                for key in ("path","mask_path"):
                    if key in asset:
                        asset[key.replace("path","name")] = Path(asset.pop(key)).name
            for key in ("image_path", "labels_path", "union_path"):
                relative = r[key]
                target = contained(root, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(contained(store.root, relative), target)
            write_json(root / "scenes" / r["capture_id"] / "current.json", r)
            from .core import png_bytes
            outline = np.zeros(scene.labels.shape,bool)
            for item in scene.active:
                x,y,w,h = item["bbox_xywh"]
                foreground = scene.labels[y:y+h,x:x+w] == item["label"]
                outline[y:y+h,x:x+w] |= foreground & ~ndimage.binary_erosion(foreground)
            atomic_bytes(root / "outlines" / (r["capture_id"] + ".png"),png_bytes(outline.astype(np.uint8)*255))
            copied.append(r)
        atomic_bytes(root / "captures.jsonl", b"".join(canonical(r) + b"\n" for r in copied))
        atomic_bytes(root / "instances.jsonl", b"".join(canonical({"capture_id": r["capture_id"], **i}) + b"\n"
                     for r in copied for i in r["instances"]))
        report = {"schema_version": PACKAGE_SCHEMA,
            "collection_id": digest(identities),
            "ready_for_training": not include_unfinished, "scene_count": len(copied),
            "instance_count": sum(i["state"] != "rejected" for r in copied for i in r["instances"]),
            "empty_count": sum(r["state"] == "confirmed_empty" for r in copied),
            "unfinished_omitted": len(records)-len(copied) if not include_unfinished else 0,
            "outline_directory": "outlines",
            "software": "multi-piece-foundation-1.0"}
        write_json(root / "collection.json", report)
        atomic_bytes(root / "README.txt", b"Native-resolution paper scene collection. Background is label 0.\n"
                     b"Arbitrary fragments are valid; no complete original page is required.\n"
                     b"Native raster outlines: outlines/<capture_id>.png. Instance labels are authoritative.\n"
                     b"Use paper-scenes import; do not combine contributor folders by overwriting files.\n")
        if check_cancelled:
            check_cancelled()
        if progress:
            progress("Rendering collection previews")
        _contact_sheet(SceneStore(root), root / "previews/contact_sheet.jpg")
        write_json(root / "checksums.json", {p.relative_to(root).as_posix(): file_hash(p)
                   for p in sorted(root.rglob("*")) if p.is_file()})
        temporary = Path(folder) / "collection.zip"
        if progress:
            progress("Compressing collection ZIP")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if check_cancelled:
                    check_cancelled()
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
        if check_cancelled:
            check_cancelled()
        os.replace(temporary, destination)
    return report


def _contact_sheet(store: SceneStore, output: Path) -> None:
    from PIL import Image, ImageDraw
    import numpy as np
    tiles = []
    for ident in store.ids()[:30]:
        scene = store.load(ident)
        image = Image.fromarray(store.rgb(scene))
        image.thumbnail((280, 240))
        labels = np.asarray(Image.fromarray(scene.labels).resize(image.size,Image.Resampling.NEAREST))
        rgb = np.array(image)
        for item in scene.active:
            pixels = labels == item["label"]
            color = np.random.default_rng(item["label"]).integers(40, 230, 3)
            rgb[pixels] = (0.6 * rgb[pixels] + 0.4 * color).astype(np.uint8)
        image = Image.fromarray(rgb)
        tile = Image.new("RGB", (300, 280), "white")
        tile.paste(image, ((300 - image.width) // 2, 35))
        ImageDraw.Draw(tile).text((8, 8), f"{len(scene.active)} pieces | {scene.record['state']}", fill="black")
        tiles.append(tile)
    if tiles:
        sheet = Image.new("RGB", (900, 280 * ((len(tiles) + 2) // 3)), "#ddd")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % 3) * 300, (i // 3) * 280))
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, quality=92)


def verify_package(root: Path) -> dict[str, Any]:
    report = read_json(root / "collection.json")
    if report.get("schema_version") != PACKAGE_SCHEMA:
        raise ValueError("Unsupported collection format")
    expected = read_json(root / "checksums.json")
    actual = {p.relative_to(root).as_posix(): file_hash(p) for p in root.rglob("*")
              if p.is_file() and p != root / "checksums.json"}
    if expected != actual:
        raise ValueError("Collection checksum inventory differs")
    store = SceneStore(root)
    store.verify(complete_only=bool(report.get("ready_for_training")))
    scenes = [store.load(i) for i in store.ids()]
    if report["scene_count"] != len(scenes) or report["collection_id"] != digest(
            [[s.record["capture_id"], annotation_key(s)] for s in scenes]):
        raise ValueError("Collection counts or annotation identity differ")
    return report


def import_collection(archive_path: Path, central: Path) -> dict[str, Any]:
    central = Path(central).resolve()
    with exclusive(central / ".import.lock"):
        journal_path = central / "pending_import.json"
        if journal_path.exists():
            pending = read_json(journal_path)
            target = contained(central,"collections/"+pending["collection_id"])
            if target.exists():
                verify_package(target)
                write_json(central/"import_reports"/(pending["collection_id"]+".json"),pending["receipt"])
                write_json(central/"capture_index.json",pending["index"])
            journal_path.unlink()
        return _import_collection(archive_path,central)


def _import_collection(archive_path: Path, central: Path) -> dict[str, Any]:
    central = Path(central).resolve()
    central.mkdir(parents=True, exist_ok=True)
    receipt = {"accepted": [], "duplicates": [], "conflicts": []}
    with tempfile.TemporaryDirectory(dir=central) as folder:
        root = Path(folder) / "package"
        root.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            total = 0
            seen = set()
            for item in archive.infolist():
                if item.is_dir():
                    continue
                target = contained(root, item.filename)
                normalized = str(target).casefold()
                if normalized in seen or (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Collection has duplicate paths or symlinks")
                seen.add(normalized)
                total += item.file_size
                if total > 20 * 1024**3:
                    raise ValueError("Collection exceeds the 20 GiB import budget")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        report = verify_package(root)
        ident = report["collection_id"]
        target = central / "collections" / ident
        if target.exists():
            verify_package(target)
            return {"collection_id": ident, "already_imported": True,
                    **read_json(central / "import_reports" / (ident + ".json"))}
        index_path = central / "capture_index.json"
        index = read_json(index_path) if index_path.exists() else {}
        store = SceneStore(root)
        for capture in store.ids():
            scene = store.load(capture)
            key = annotation_key(scene)
            previous = index.get(capture)
            state = "accepted" if previous is None else "duplicates" if previous["annotation_key"] == key else "conflicts"
            if state == "accepted":
                receipt["accepted"].append(capture)
                index[capture] = {"collection_id": ident, "annotation_key": key}
            else:
                receipt[state].append(capture)
        # Import receipts are internal; keep the original checked package untouched.
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(central / "pending_import.json", {"collection_id":ident,"receipt":receipt,"index":index})
        os.replace(root, target)
        write_json(central / "import_reports" / (ident + ".json"), receipt)
        write_json(index_path, index)
        (central / "pending_import.json").unlink()
    return {"collection_id": ident, "already_imported": False, **receipt}


def resolve_annotation(central: Path, capture_id: str, collection_id: str) -> dict[str,Any]:
    """Explicitly choose a preserved annotation revision; never mutate a package/dataset."""
    from .core import ID_PATTERN
    if not ID_PATTERN.fullmatch(capture_id) or not ID_PATTERN.fullmatch(collection_id):
        raise ValueError("Invalid capture or collection ID")
    central = Path(central).resolve()
    with exclusive(central/".import.lock"):
        package = central/"collections"/collection_id
        verify_package(package)
        scene = SceneStore(package).load(capture_id)
        validate_scene(scene,require_complete=True)
        path = central/"capture_index.json"
        index = read_json(path)
        previous = index.get(capture_id)
        if previous is None:
            raise ValueError("Capture is not registered centrally")
        selected = {"collection_id":collection_id,"annotation_key":annotation_key(scene)}
        receipt = {"capture_id":capture_id,"previous":previous,"selected":selected}
        write_json(central/"resolutions"/(digest(receipt)+".json"),receipt)
        index[capture_id] = selected
        write_json(path,index)
    return receipt
