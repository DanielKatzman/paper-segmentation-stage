"""Persistent, background SegFormer proposal cache for annotation."""

from __future__ import annotations

import heapq
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from universal_segmenter.inference.preprocess import load_rgb
from universal_segmenter.inference.release import ReleaseConfig, Segmenter, load_release


CACHE_SCHEMA = "paper-segmentation-proposal-cache-1.0"
MODEL_KEY = "segformer_b1_detail"


def _source_hash(record: dict[str, Any]) -> str:
    value = str(record.get("source_sha256", "")).lower()
    if re.fullmatch(r"[0-9a-f]{64}", value):
        return value
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(
        temporary, format="PNG", compress_level=1)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class PortableModel:
    release: ReleaseConfig
    key: str = MODEL_KEY
    display_name: str = "SegFormer-B1 Detail v3"

    @property
    def model_type(self) -> str:
        return MODEL_KEY

    @property
    def resolution(self) -> int:
        return self.release.resolution

    @property
    def threshold(self) -> float:
        return self.release.threshold

    @property
    def model_path(self) -> Path:
        return self.release.model_path

    @property
    def metadata_path(self) -> Path:
        return self.release.release_path

    @property
    def model_sha256(self) -> str:
        return self.release.model_sha256

    @property
    def cache_namespace(self) -> str:
        return self.release.cache_namespace


def load_portable_model(project: Path | None = None,
                        require_holdout_pass: bool = True) -> PortableModel:
    return PortableModel(load_release(
        project, require_holdout_pass=require_holdout_pass))


def load_portable_models(project: Path | None = None,
                         require_all: bool = True) -> list[PortableModel]:
    """Compatibility helper: this release intentionally returns one model."""
    del require_all
    return [load_portable_model(project)]


class PredictionCache:
    """Compute one immutable SegFormer guess at a time in the background."""

    def __init__(self, project: Path, cache_root: Path, model: PortableModel,
                 cpu_threads: int, pdf_dpi: int = 300,
                 image_loader: Callable[..., np.ndarray] = load_rgb) -> None:
        self.project = Path(project).resolve()
        self.cache_root = Path(cache_root).resolve()
        self.model = model
        self.cpu_threads = max(1, int(cpu_threads))
        self.pdf_dpi = int(pdf_dpi)
        self.image_loader = image_loader
        self._condition = threading.Condition()
        self._queue: list[tuple[int, int, str]] = []
        self._priorities: dict[str, int] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._ready: set[str] = set()
        self._running: set[str] = set()
        self._errors: dict[str, str] = {}
        self._sequence = 0
        self._total = 0
        self._stop = False
        self._worker: threading.Thread | None = None
        self._segmenter: Segmenter | None = None
        self._generated = 0
        self._reused = 0
        self._started_utc = datetime.now(timezone.utc).isoformat()
        _atomic_json(self._namespace / "model.json", {
            "schema_version": CACHE_SCHEMA,
            "model_key": model.key,
            "display_name": model.display_name,
            "resolution": model.resolution,
            "threshold": model.threshold,
            "postprocess_policy": model.release.postprocess_policy,
            "onnx_sha256": model.model_sha256,
        })

    @property
    def _namespace(self) -> Path:
        return self.cache_root / "predictions" / self.model.cache_namespace

    def _source_path(self, record: dict[str, Any]) -> Path:
        value = Path(str(record["image_path"]))
        return value if value.is_absolute() else self.project / value

    def _cache_path(self, record: dict[str, Any]) -> Path:
        return self._namespace / f"{_source_hash(record)}.png"

    @staticmethod
    def _read_mask(path: Path,
                   expected_shape: tuple[int, int] | None = None) -> np.ndarray | None:
        if not path.is_file():
            return None
        try:
            with Image.open(path) as source:
                mask = np.asarray(source.convert("L"), dtype=np.uint8) >= 128
        except (OSError, ValueError):
            return None
        if expected_shape is not None and mask.shape != expected_shape:
            return None
        return mask

    def _schedule(self, source_hash: str, priority: int) -> None:
        if source_hash in self._ready or source_hash in self._running:
            return
        current = self._priorities.get(source_hash)
        if current is not None and current <= priority:
            return
        self._priorities[source_hash] = priority
        self._sequence += 1
        heapq.heappush(self._queue, (priority, self._sequence, source_hash))
        self._condition.notify_all()

    def start(self, records: Sequence[dict[str, Any]]) -> None:
        unique: list[str] = []
        for record in records:
            source_hash = _source_hash(record)
            if source_hash not in self._records:
                self._records[source_hash] = record
                unique.append(source_hash)
        self._total = len(unique)
        with self._condition:
            if self._worker is not None:
                raise RuntimeError("prediction cache worker was already started")
            for index, source_hash in enumerate(unique):
                if self._cache_path(self._records[source_hash]).is_file():
                    self._ready.add(source_hash)
                    self._reused += 1
                else:
                    self._schedule(source_hash, index + 1000)
            self._worker = threading.Thread(
                target=self._work_loop, name="segformer-proposal-cache", daemon=True)
            self._worker.start()

    def _predict(self, record: dict[str, Any]) -> None:
        rgb = self.image_loader(self._source_path(record), pdf_dpi=self.pdf_dpi)
        path = self._cache_path(record)
        if self._read_mask(path, rgb.shape[:2]) is not None:
            return
        if self._segmenter is None:
            self._segmenter = Segmenter(
                cpu_threads=self.cpu_threads, release=self.model.release)
        result = self._segmenter.segment(rgb)
        _atomic_mask(path, result.mask)
        self._generated += 1

    def _work_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                priority, _, source_hash = heapq.heappop(self._queue)
                if self._priorities.get(source_hash) != priority:
                    continue
                self._priorities.pop(source_hash, None)
                self._running.add(source_hash)
            try:
                self._predict(self._records[source_hash])
            except Exception as error:
                with self._condition:
                    self._errors[source_hash] = f"{type(error).__name__}: {error}"
            finally:
                with self._condition:
                    self._running.discard(source_hash)
                    if source_hash not in self._errors:
                        self._ready.add(source_hash)
                    self._condition.notify_all()

    def get(self, record: dict[str, Any],
            rgb: np.ndarray | None = None) -> np.ndarray:
        source_hash = _source_hash(record)
        expected_shape = None if rgb is None else rgb.shape[:2]
        path = self._cache_path(record)
        cached = self._read_mask(path, expected_shape)
        if cached is not None:
            return cached
        with self._condition:
            self._records[source_hash] = record
            self._errors.pop(source_hash, None)
            self._schedule(source_hash, -100)
            while source_hash not in self._ready and source_hash not in self._errors:
                self._condition.wait()
            if source_hash in self._errors:
                raise RuntimeError(
                    f"SegFormer prediction failed: {self._errors[source_hash]}")
        cached = self._read_mask(path, expected_shape)
        if cached is None:
            raise RuntimeError(f"prediction cache is missing or invalid: {path}")
        return cached

    def status_text(self) -> str:
        with self._condition:
            return (
                f"{self.model.display_name}: {len(self._ready)}/{self._total} cached"
                + (f", {len(self._errors)} failed" if self._errors else ""))

    def wait_until_idle(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._priorities or self._running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, wait_seconds: float = 5.0) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=max(0.0, wait_seconds))
        _atomic_json(self.cache_root / "prediction_cache_report.json", {
            "schema_version": CACHE_SCHEMA,
            "started_utc": self._started_utc,
            "closed_utc": datetime.now(timezone.utc).isoformat(),
            "source_count": self._total,
            "generated": self._generated,
            "reused_at_start": self._reused,
            "ready": len(self._ready),
            "errors": dict(sorted(self._errors.items())),
        })


class MLProposalProvider:
    """Adapter consumed by the correction editor."""

    primary_model = MODEL_KEY

    def __init__(self, cache: PredictionCache) -> None:
        self.cache = cache
        self.choices = [(MODEL_KEY, cache.model.display_name)]

    def get(self, record: dict[str, Any], rgb: np.ndarray,
            key: str) -> tuple[np.ndarray, str]:
        if key != MODEL_KEY:
            raise ValueError(f"unknown proposal model: {key}")
        model = self.cache.model
        return (
            self.cache.get(record, rgb),
            f"onnx:{model.model_type}:{model.model_sha256[:16]}:"
            f"{model.resolution}:t{model.threshold:g}:"
            f"{model.release.postprocess_policy}",
        )

    def status_text(self) -> str:
        return self.cache.status_text()

