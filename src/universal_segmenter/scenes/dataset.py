"""Central grouped splits, immutable dataset assembly and native scene loading."""
from __future__ import annotations

import csv
import io
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .core import (Scene, SceneStore, SPLITS, atomic_bytes, contained, digest, file_hash,
                   exclusive, read_json, write_json)
from .exchange import annotation_key

DATASET_SCHEMA = "paper-scene-dataset-1.0"


def group_tokens(scene: Scene) -> list[str]:
    m = scene.record["metadata"]
    result = ["capture:" + scene.record["pixel_sha256"]]
    if m.get("split_group_id"):
        result.append("group:" + str(m["split_group_id"]))
    if m.get("document_id") and m.get("page_id"):
        result.append("page:" + digest([m["document_id"], m["page_id"]]))
    if m.get("capture_session_id"):
        result.append("session:" + str(m["capture_session_id"]))
    for item in scene.active:
        if item.get("physical_piece_id"):
            result.append("physical:" + digest([m.get("document_id"), item["physical_piece_id"]]))
    return sorted(set(result))


def assign_splits(scenes: list[Scene], ledger_path: Path, *, seed: int = 73191,
                  ratios: dict[str,float] | None = None,
                  protected_tokens: dict[str,str] | None = None) -> dict[str,str]:
    ledger_path = Path(ledger_path)
    with exclusive(ledger_path.with_suffix(".lock")):
        return _assign_splits(scenes,ledger_path,seed=seed,ratios=ratios,protected_tokens=protected_tokens)


def _assign_splits(scenes: list[Scene], ledger_path: Path, *, seed: int,
                   ratios: dict[str,float] | None, protected_tokens: dict[str,str] | None) -> dict[str,str]:
    ratios = ratios or {"train": .70, "validation": .10, "calibration": .10, "holdout": .10}
    if set(ratios) != SPLITS or min(ratios.values()) < 0 or sum(ratios.values()) <= 0:
        raise ValueError("Provide nonnegative weights for all four dataset splits")
    ledger = read_json(ledger_path) if ledger_path.exists() else {
        "schema_version": "paper-scene-splits-1.0", "seed": seed, "ratios": ratios, "tokens": {}}
    if ledger.get("schema_version") != "paper-scene-splits-1.0" or ledger["seed"] != seed or ledger["ratios"] != ratios:
        raise ValueError("Split policy changed; use a new ledger instead of rewriting frozen assignments")
    tokens = dict(ledger["tokens"])
    for token, split in (protected_tokens or {}).items():
        if token in tokens and tokens[token] != split:
            raise ValueError("Protected legacy split conflicts with central assignment")
        tokens[token] = split
    # Connected components of shared page, session, source and physical identity.
    parent: dict[str,str] = {}
    def find(x):
        parent.setdefault(x,x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for scene in scenes:
        group = group_tokens(scene)
        for token in group:
            parent[find(token)] = find(group[0])
    groups: dict[str,list[Scene]] = {}
    for scene in scenes:
        groups.setdefault(find(group_tokens(scene)[0]), []).append(scene)
    result = {}
    split_names = sorted(ratios)
    for _, members in sorted(groups.items()):
        keys = sorted({t for s in members for t in group_tokens(s)})
        existing = {tokens[t] for t in keys if t in tokens}
        existing.update(s.record["metadata"]["split"] for s in members if s.record["metadata"].get("split") in SPLITS)
        if len(existing) > 1:
            raise ValueError("Related captures cross protected splits; quarantine/reconcile the collection")
        if existing:
            split = existing.pop()
            if split not in SPLITS:
                raise ValueError("Related page is protected by multiple legacy evaluation splits")
        else:
            value = int(digest([seed,keys])[:13],16) / 16**13
            accumulated = 0.
            split = split_names[-1]
            for candidate in split_names:
                accumulated += ratios[candidate] / sum(ratios.values())
                if value < accumulated:
                    split = candidate
                    break
        tokens.update({t:split for t in keys})
        result.update({s.record["capture_id"]:split for s in members})
    write_json(ledger_path, {**ledger, "tokens": tokens})
    return result


def _legacy_rows(root: Path):
    with (root / "samples.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _protected_legacy(roots: list[Path]) -> dict[str,str]:
    tokens = {}
    for root in roots:
        for row in _legacy_rows(root):
            if row["split"] == "train":
                continue
            # Old datasets may split pieces of one page; never assign new scenes
            # from that page to training based only on an unseen pixel hash.
            if row.get("document_id") and row.get("page_id"):
                key = "page:" + digest([row["document_id"],row["page_id"]])
                if key in tokens and tokens[key] != row["split"]:
                    tokens[key] = "protected"
                else:
                    tokens[key] = row["split"]
            if row.get("sample_kind") == "real_capture":
                from .core import pixel_hash
                with Image.open(root / row["image_path"]) as image:
                    tokens["capture:"+pixel_hash(np.asarray(image.convert("RGB"),np.uint8))] = row["split"]
    return tokens


def build_dataset(roots: list[Path], output: Path, ledger_path: Path, *, seed: int = 73191,
                  ratios: dict[str,float] | None = None, legacy_roots: list[Path] | None = None,
                  legacy_exhaustive_ids: set[str] | None = None, central: Path | None = None) -> dict[str, Any]:
    output = Path(output).resolve()
    legacy_roots = [Path(p).resolve() for p in (legacy_roots or [])]
    stores = [SceneStore(p) for p in roots]
    if Path(ledger_path).resolve().is_relative_to(output):
        raise ValueError("The mutable split ledger must remain outside a frozen dataset")
    if any(output == root or output.is_relative_to(root) or root.is_relative_to(output) for root in legacy_roots):
        raise ValueError("Dataset output must not overlap a preserved legacy dataset")
    selections = [(store,None) for store in stores]
    if central:
        from .exchange import verify_package
        central = Path(central).resolve()
        index = read_json(central/"capture_index.json")
        by_collection: dict[str,list[str]] = {}
        for capture,item in index.items():
            by_collection.setdefault(item["collection_id"],[]).append(capture)
        for collection,identities in by_collection.items():
            package = contained(central,"collections/"+collection)
            verify_package(package)
            store = SceneStore(package)
            for ident in identities:
                if annotation_key(store.load(ident))!=index[ident]["annotation_key"]:
                    raise ValueError("Central annotation selection identity differs")
            stores.append(store)
            selections.append((store,identities))
    if output.exists() or any(output.is_relative_to(s.root) or s.root.is_relative_to(output) for s in stores):
        raise ValueError("Dataset must be new and separate from source collections")
    sources = {}
    unfinished = 0
    for store,identities in selections:
        store.verify()
        for ident in (identities if identities is not None else store.ids()):
            scene = store.load(ident)
            if not scene.eligible:
                unfinished += 1
                continue
            previous = sources.get(ident)
            if previous and annotation_key(previous[1]) != annotation_key(scene):
                raise ValueError("Conflicting annotations for the same capture; reconcile before freezing")
            sources.setdefault(ident, (store,scene))
    protected = _protected_legacy(legacy_roots)
    for _,scene in sources.values():
        if any(protected.get(t) == "protected" for t in group_tokens(scene)):
            raise ValueError("Page already spans legacy protected splits; use independent new pages")
    real = [s for _,s in sources.values() if not s.record["metadata"].get("sample_kind", "").startswith("synthetic_")
            and s.record["metadata"].get("sample_kind") != "background_only"]
    assignments = assign_splits(real, Path(ledger_path), seed=seed, ratios=ratios,
                                protected_tokens=protected)
    if not sources and not legacy_exhaustive_ids:
        raise ValueError("No fully annotated scenes were selected")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as folder:
        target = Path(folder) / "dataset"
        store_out = SceneStore(target)
        records, input_receipts = [], []
        for ident,(store,scene) in sorted(sources.items()):
            m = scene.record["metadata"]
            generated = m.get("sample_kind", "").startswith("synthetic_") or m.get("sample_kind") == "background_only"
            split = "train" if generated else assignments[ident]
            if generated:
                recipe = m.get("recipe", {})
                if not recipe.get("sources") or any(a.get("split") != "train" for a in recipe["sources"]):
                    raise ValueError("Generated samples lack training-only source lineage")
            imported = store_out.add(store.rgb(scene), m)
            imported.labels = scene.labels.copy()
            imported.record = {**scene.record, "image_path": imported.record["image_path"],
                               "image_sha256": imported.record["image_sha256"],
                               "metadata": {**m,"split":split}, "revision": imported.record["revision"]}
            store_out.save(imported)
            kind = m.get("sample_kind") if generated else "real_scene_empty" if not scene.active else "real_scene"
            records.append({"sample_id":ident, "split":split, "sample_kind":kind,
                "capture_profile":m.get("capture_profile","phone"), "group_id":digest(group_tokens(scene)),
                "image_path":imported.record["image_path"], "mask_path":imported.record["union_path"],
                "labels_path":imported.record["labels_path"], "scene_path":f"scenes/{ident}/current.json",
                "instance_count":len(scene.active), "document_id":m.get("document_id",""),
                "page_id":m.get("page_id",""), "source_revision":scene.record["revision"]})
            input_receipts.append({"capture_id":ident,"revision":scene.record["revision"],"annotation_key":annotation_key(scene)})
        omitted_legacy = 0
        for legacy in legacy_roots:
            for row in _legacy_rows(legacy):
                if row["split"] != "train" or row["sample_id"] not in (legacy_exhaustive_ids or set()):
                    omitted_legacy += 1
                    continue
                new_row = dict(row)
                if any(r["sample_id"]==row["sample_id"] for r in records):
                    raise ValueError("Duplicate sample ID across selected legacy sources")
                with Image.open(contained(legacy,row["image_path"])) as image:
                    image_hw = (image.height,image.width)
                with Image.open(contained(legacy,row["mask_path"])) as image:
                    mask = np.asarray(image.convert("L"))
                if mask.shape!=image_hw or not set(np.unique(mask)).issubset({0,255}):
                    raise ValueError("Legacy image/mask dimensions or binary values are invalid")
                for key in ("image_path","mask_path"):
                    src = contained(legacy,row[key])
                    relative = "legacy/"+digest(str(legacy))[:12]+"/"+row[key]
                    dst = contained(target,relative)
                    dst.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(src,dst)
                    new_row[key] = relative
                records.append(new_row)
        fields = sorted({k for r in records for k in r})
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        atomic_bytes(target/"samples.csv",stream.getvalue().encode())
        counts = dict(Counter(r["split"] for r in records))
        report = {"schema_version":DATASET_SCHEMA,"valid":True,"sample_count":len(records),
            "samples_by_split":counts,"samples_by_kind":dict(Counter(r["sample_kind"] for r in records)),
            "unfinished_omitted":unfinished,"legacy_omitted":omitted_legacy,
            "evaluation_ready":all(any(r["split"]==s and r["sample_kind"] in {"real_scene","real_scene_empty"}
                                       for r in records) for s in SPLITS), "errors":[]}
        write_json(target/"dataset_report.json",report)
        write_json(target/"source_receipt.json",input_receipts)
        write_json(target/"split_ledger.json",read_json(Path(ledger_path)))
        artifacts = {p.relative_to(target).as_posix():file_hash(p) for p in sorted(target.rglob("*")) if p.is_file()}
        write_json(target/"scene_dataset_freeze.json",{"schema_version":DATASET_SCHEMA,
                   "artifacts":artifacts,"fingerprint":digest(artifacts),"report":report})
        os.replace(target,output)
    return report


def verify_dataset(root: Path) -> dict[str,Any]:
    root = Path(root).resolve()
    freeze = read_json(root/"scene_dataset_freeze.json")
    actual = {p.relative_to(root).as_posix():file_hash(p) for p in sorted(root.rglob("*"))
              if p.is_file() and p.name != "scene_dataset_freeze.json"}
    if freeze.get("schema_version") != DATASET_SCHEMA or actual != freeze["artifacts"] or digest(actual) != freeze["fingerprint"]:
        raise ValueError("Frozen dataset artifacts changed")
    SceneStore(root).verify(complete_only=True)
    return freeze["report"]


def load_view(root: Path, row: dict[str,Any], resolution: int,
              tile_xywh: tuple[int,int,int,int] | None = None):
    from universal_segmenter.inference.preprocess import letterbox
    if resolution < 16:
        raise ValueError("Resolution must be at least 16")
    root = Path(root)
    with Image.open(contained(root,row["image_path"])) as image:
        rgb = np.asarray(image.convert("RGB"),np.uint8)
    if row.get("labels_path"):
        with np.load(contained(root,row["labels_path"]),allow_pickle=False) as arrays:
            labels = arrays["labels"].astype(np.int32)
    else:
        with Image.open(contained(root,row["mask_path"])) as image:
            labels = ndimage.label(np.asarray(image.convert("L"))>=128)[0].astype(np.int32)
    if labels.shape != rgb.shape[:2]:
        raise ValueError("Training image and instance-label dimensions differ")
    if tile_xywh:
        x,y,w,h = tile_xywh
        if min(x,y)<0 or min(w,h)<1 or x+w>rgb.shape[1] or y+h>rgb.shape[0]:
            raise ValueError("Tile extends outside source")
        rgb,labels = rgb[y:y+h,x:x+w],labels[y:y+h,x:x+w]
    boxed,transform = letterbox(rgb,resolution)
    transformed = np.asarray(Image.fromarray(labels).resize((transform.resized_width,transform.resized_height),
                             Image.Resampling.NEAREST),np.int32)
    target = np.zeros((resolution,resolution),np.int32)
    target[transform.top:transform.top+transform.resized_height,
           transform.left:transform.left+transform.resized_width] = transformed
    return boxed,target,{"letterbox": {"left":transform.left,"top":transform.top,
        "resized_width":transform.resized_width,"resized_height":transform.resized_height},
        "tile_xywh":tile_xywh,"original_hw":list(rgb.shape[:2])}


def instance_metrics(prediction: np.ndarray, target: np.ndarray, threshold: float = .5) -> dict[str,Any]:
    if prediction.shape != target.shape:
        raise ValueError("Instance maps must have matching shapes")
    pids,tids = [int(i) for i in np.unique(prediction) if i],[int(i) for i in np.unique(target) if i]
    ious = np.zeros((len(pids),len(tids)),float)
    for i,p in enumerate(pids):
        for j,t in enumerate(tids):
            a,b = prediction==p,target==t
            ious[i,j] = (a&b).sum()/max(1,(a|b).sum())
    pi,ti = linear_sum_assignment(-ious)
    matches = int((ious[pi,ti]>=threshold).sum())
    return {"predicted_count":len(pids),"target_count":len(tids),"matched_count":matches,
        "precision":matches/len(pids) if pids else float(not tids),
        "recall":matches/len(tids) if tids else 1.,"exact_count":len(pids)==len(tids),
        "matched_ious":ious[pi,ti].tolist()}
