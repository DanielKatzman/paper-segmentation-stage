"""Versioned scene records and atomic, revisioned native-resolution storage."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

SCHEMA = "paper-scene-1.0"
STATES = {"in_progress", "complete", "confirmed_empty", "excluded"}
SPLITS = {"train", "validation", "calibration", "holdout"}
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def contained(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Artifact path must be a nonempty portable relative path")
    path = Path(relative)
    target = (Path(root) / path).resolve()
    if (path.is_absolute() or ":" in relative or ".." in path.parts
            or path.as_posix() != relative or not target.is_relative_to(Path(root).resolve())):
        raise ValueError("Artifact path escapes its collection")
    return target


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def exclusive(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError(f"Operation is locked: {path}. Check the recorded process before recovering a stale lock.") from error
    try:
        with os.fdopen(fd,"w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def png_bytes(array: np.ndarray) -> bytes:
    import io
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG", compress_level=1)
    return stream.getvalue()


def pixel_hash(rgb: np.ndarray) -> str:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Source must be HxWx3 uint8 RGB")
    sha = hashlib.sha256(canonical(list(rgb.shape)))
    sha.update(memoryview(np.ascontiguousarray(rgb)))
    return sha.hexdigest()


def bbox(mask: np.ndarray) -> list[int]:
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("An instance cannot have an empty mask")
    return [int(x.min()), int(y.min()), int(x.max() - x.min() + 1), int(y.max() - y.min() + 1)]


def instance_errors(mask: np.ndarray) -> list[str]:
    errors = []
    if not mask.any() or mask.all():
        errors.append("empty_or_full")
    if ndimage.label(mask)[1] != 1:
        errors.append("disconnected")
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        errors.append("clipped")
    return errors


@dataclass
class Scene:
    record: dict[str, Any]
    labels: np.ndarray

    def copy(self) -> "Scene":
        return Scene(copy.deepcopy(self.record), self.labels.copy())

    @property
    def active(self) -> list[dict[str, Any]]:
        return [i for i in self.record["instances"] if i["state"] != "rejected"]

    @property
    def eligible(self) -> bool:
        return self.record["state"] in {"complete", "confirmed_empty"}


def new_scene(rgb: np.ndarray, metadata: dict[str, Any] | None = None) -> Scene:
    sha = pixel_hash(rgb)
    metadata = copy.deepcopy(metadata or {})
    return Scene({"schema_version": SCHEMA, "capture_id": "source_" + sha,
                  "pixel_sha256": sha, "original_size_hw": list(rgb.shape[:2]),
                  "state": "in_progress", "instances": [], "metadata": metadata,
                  "proposal": None, "clean_background": False},
                 np.zeros(rgb.shape[:2], dtype=np.int32))


def validate_scene(scene: Scene, *, require_complete: bool = False) -> None:
    r, labels = scene.record, scene.labels
    if r.get("schema_version") != SCHEMA or not ID_PATTERN.fullmatch(r.get("capture_id", "")):
        raise ValueError("Invalid scene schema or capture ID")
    if labels.ndim != 2 or labels.dtype.kind not in "iu" or labels.min(initial=0) < 0:
        raise ValueError("Labels must be a nonnegative integer matrix")
    if list(labels.shape) != r.get("original_size_hw") or min(labels.shape) < 2:
        raise ValueError("Label dimensions differ from source dimensions")
    if r.get("state") not in STATES or not isinstance(r.get("metadata"), dict):
        raise ValueError("Invalid scene state or metadata")
    if r.get("capture_id") != "source_" + str(r.get("pixel_sha256", "")):
        raise ValueError("Capture ID must match normalized source identity")
    if require_complete and not scene.eligible:
        raise ValueError("Scene is not exhaustively reviewed or confirmed empty")
    if scene.eligible:
        meta = r["metadata"]
        if meta.get("capture_profile") not in {"phone", "scanner"}:
            raise ValueError("Complete scenes need capture type (phone or scanner)")
    ids, values, all_values = set(), set(), set()
    unique, counts = np.unique(labels, return_counts=True)
    areas = dict(zip(unique.tolist(), counts.tolist()))
    for item in r["instances"]:
        ident, label = item.get("instance_id", ""), item.get("label")
        if (not ID_PATTERN.fullmatch(ident) or ident in ids or type(label) is not int
                or not 0 < label <= np.iinfo(np.int32).max or label in all_values):
            raise ValueError("Instance IDs and positive labels must be unique")
        ids.add(ident)
        all_values.add(label)
        if item.get("state") not in {"proposed", "accepted", "rejected"}:
            raise ValueError("Invalid instance review state")
        if item["state"] == "rejected":
            if areas.get(label, 0):
                raise ValueError("Rejected instance still owns foreground")
        else:
            values.add(label)
            box = item.get("bbox_xywh")
            if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                    or min(box[:2]) < 0 or min(box[2:]) < 1
                    or box[0] + box[2] > labels.shape[1] or box[1] + box[3] > labels.shape[0]
                    or not areas.get(label, 0)):
                raise ValueError("Instance mask is empty or bbox is stale")
            x, y, w, h = box
            mask = labels[y:y+h, x:x+w] == label
            if int(mask.sum()) != areas[label] or bbox(mask) != [0, 0, w, h]:
                raise ValueError("Instance mask is empty or bbox is stale")
            if item["state"] == "accepted":
                errors = []
                if areas[label] == labels.size:
                    errors.append("empty_or_full")
                if ndimage.label(mask)[1] != 1:
                    errors.append("disconnected")
                if x == 0 or y == 0 or x+w == labels.shape[1] or y+h == labels.shape[0]:
                    errors.append("clipped")
                if errors:
                    raise ValueError(f"Accepted instance is invalid: {errors}")
    actual = set(areas) - {0}
    if actual != values:
        raise ValueError("Label map and instance records disagree")
    if r["state"] == "complete" and (not values or any(i["state"] != "accepted" for i in scene.active)):
        raise ValueError("Complete scenes need every instance accepted")
    if r["state"] == "confirmed_empty" and (values or not r.get("empty_confirmed")):
        raise ValueError("An empty scene needs an explicit empty confirmation")
    if r.get("clean_background") and r["state"] != "confirmed_empty":
        raise ValueError("Only a confirmed-empty scene can be a clean background")


class SceneStore:
    """One native image per capture; immutable revisions and atomic current pointer."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def ids(self) -> list[str]:
        return sorted(p.parent.name for p in (self.root / "scenes").glob("*/current.json"))

    def ensure_writable(self) -> None:
        if (self.root / "checksums.json").exists() or (self.root / "scene_dataset_freeze.json").exists():
            raise ValueError("Checked collections and frozen datasets are immutable; clone-workspace before editing")

    def add(self, rgb: np.ndarray, metadata: dict[str, Any] | None = None) -> Scene:
        self.ensure_writable()
        scene = new_scene(rgb, metadata)
        ident = scene.record["capture_id"]
        if (self.root / "scenes" / ident / "current.json").exists():
            return self.load(ident)
        image_path = f"images/{ident}.png"
        atomic_bytes(self.root / image_path, png_bytes(rgb))
        scene.record.update(image_path=image_path, image_sha256=file_hash(self.root / image_path))
        self.save(scene)
        return scene

    def record(self, ident: str) -> dict[str, Any]:
        """Read the small current record without decoding native masks or images."""
        if not ID_PATTERN.fullmatch(ident):
            raise ValueError("Unsafe capture ID")
        return read_json(self.root / "scenes" / ident / "current.json")

    def load(self, ident: str) -> Scene:
        record = self.record(ident)
        path = contained(self.root, record["labels_path"])
        if file_hash(path) != record["labels_sha256"]:
            raise ValueError("Annotation label checksum mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            labels = arrays["labels"].copy()
        scene = Scene(record, labels)
        validate_scene(scene)
        return scene

    def rgb(self, scene: Scene) -> np.ndarray:
        path = contained(self.root, scene.record["image_path"])
        if file_hash(path) != scene.record["image_sha256"]:
            raise ValueError("Source image checksum mismatch")
        with Image.open(path) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        if pixel_hash(rgb) != scene.record["pixel_sha256"]:
            raise ValueError("Decoded source identity mismatch")
        return rgb

    def save(self, scene: Scene, *, expected_revision: str | None = None) -> None:
        self.ensure_writable()
        with exclusive(self.root / "scenes" / scene.record["capture_id"] / ".write.lock"):
            self._save(scene, expected_revision=expected_revision)

    def _save(self, scene: Scene, *, expected_revision: str | None = None) -> None:
        import io
        validate_scene(scene)
        r = copy.deepcopy(scene.record)
        folder = self.root / "scenes" / r["capture_id"]
        current = folder / "current.json"
        if current.exists() and read_json(current).get("revision") != (
                expected_revision if expected_revision is not None else r.get("revision")):
            raise ValueError("Scene changed since it was loaded; reload before editing")
        parent = r.pop("revision", None)
        if expected_revision is not None:
            parent = expected_revision
        for key in ("labels_path", "labels_sha256", "union_path", "union_sha256", "parent_revision"):
            r.pop(key, None)
        revision = digest([r, hashlib.sha256(scene.labels.astype("<i4").tobytes()).hexdigest(), parent])
        stem = f"scenes/{r['capture_id']}/{revision}"
        stream = io.BytesIO()
        np.savez_compressed(stream, labels=scene.labels.astype(np.int32))
        atomic_bytes(self.root / (stem + ".npz"), stream.getvalue())
        atomic_bytes(self.root / (stem + ".png"), png_bytes((scene.labels > 0).astype(np.uint8) * 255))
        r.update(revision=revision, parent_revision=parent, labels_path=stem + ".npz",
                 labels_sha256=file_hash(self.root / (stem + ".npz")), union_path=stem + ".png",
                 union_sha256=file_hash(self.root / (stem + ".png")))
        write_json(self.root / (stem + ".json"), r)
        write_json(current, r)
        scene.record = r

    def restore(self, ident: str, revision: str) -> Scene:
        current = self.load(ident)
        if not re.fullmatch(r"[a-f0-9]{64}", revision):
            raise ValueError("Invalid revision")
        r = read_json(self.root / "scenes" / ident / (revision + ".json"))
        with np.load(contained(self.root, r["labels_path"]), allow_pickle=False) as arrays:
            restored = Scene(r, arrays["labels"].copy())
        self.save(restored, expected_revision=current.record["revision"])
        return restored

    def verify(self, *, complete_only: bool = False) -> dict[str, int]:
        counts = {"scenes": 0, "complete": 0, "empty": 0, "unfinished": 0, "excluded": 0, "instances": 0}
        for ident in self.ids():
            scene = self.load(ident)
            self.rgb(scene)
            with Image.open(contained(self.root, scene.record["union_path"])) as image:
                union = np.asarray(image)
            if (file_hash(contained(self.root, scene.record["union_path"])) != scene.record["union_sha256"]
                    or not np.array_equal(union, (scene.labels > 0).astype(np.uint8) * 255)):
                raise ValueError("Union mask does not match instance labels")
            if complete_only:
                validate_scene(scene, require_complete=True)
            counts["scenes"] += 1
            counts["instances"] += len(scene.active)
            counts[{"complete": "complete", "confirmed_empty": "empty", "in_progress": "unfinished", "excluded": "excluded"}[scene.record["state"]]] += 1
        return counts
