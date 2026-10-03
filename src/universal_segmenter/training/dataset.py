"""Generated-dataset loader and deterministic domain-balanced epoch plans."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from PIL import Image

from universal_segmenter.losses import signed_distance_target

from .freeze import read_sample_rows


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def sample_category(sample_kind: str) -> str:
    if sample_kind == "real_capture":
        return "real_capture"
    if sample_kind in {"real_piece_composite", "real_shape_retextured"}:
        return "real_derived"
    if sample_kind == "procedural_piece":
        return "procedural"
    if sample_kind == "background_only":
        return "background"
    raise ValueError(f"unsupported sample kind: {sample_kind}")


def _seed(*values: object) -> int:
    payload = "\0".join(map(str, values)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def deterministic_seed(*values: object) -> int:
    """Public stable seed helper for isolated loader generators."""
    return _seed(*values)


@dataclass(frozen=True, slots=True)
class SampleDraw:
    row_index: int
    augmentation_seed: int


def _allocate(total: int, weights: dict[str, float],
              available: set[str]) -> dict[str, int]:
    effective = {name: value for name, value in weights.items()
                 if name in available and value > 0.0}
    if not effective:
        raise ValueError("no positively weighted training categories are available")
    denominator = sum(effective.values())
    exact = {name: total * value / denominator for name, value in effective.items()}
    result = {name: math.floor(value) for name, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(effective, key=lambda name: (-(exact[name] - result[name]), name))
    for name in order[:remaining]:
        result[name] += 1
    return result


def _balanced_profile_draws(indices: list[int], rows: Sequence[dict[str, str]],
                            count: int, rng: np.random.Generator) -> list[int]:
    by_profile: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_profile[rows[index]["capture_profile"]].append(index)
    profiles = sorted(by_profile)
    if len(profiles) <= 1:
        return rng.choice(indices, size=count, replace=True).tolist()
    draws: list[int] = []
    for draw_index in range(count):
        profile = profiles[draw_index % len(profiles)]
        candidates = by_profile[profile]
        draws.append(int(candidates[int(rng.integers(0, len(candidates)))]))
    return draws


def build_epoch_plan(rows: Sequence[dict[str, str]], epoch: int,
                     seed: int, total_samples: int,
                     category_weights: dict[str, float]) -> list[SampleDraw]:
    """Create the complete deterministic draw order for one training epoch."""
    if total_samples < 1:
        raise ValueError("total_samples must be positive")
    by_category: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == "train":
            by_category[sample_category(row["sample_kind"])].append(index)
    counts = _allocate(total_samples, category_weights, set(by_category))
    rng = np.random.default_rng(_seed(seed, epoch, "epoch_plan"))
    chosen: list[int] = []
    for category, count in sorted(counts.items()):
        indices = by_category[category]
        if category == "real_capture":
            chosen.extend(_balanced_profile_draws(indices, rows, count, rng))
        else:
            chosen.extend(rng.choice(indices, size=count, replace=True).tolist())
    rng.shuffle(chosen)
    return [SampleDraw(
        int(index), _seed(seed, epoch, position, rows[int(index)]["sample_id"], "aug"))
        for position, index in enumerate(chosen)]


class DrawSampler:
    """A sampler over an already materialized deterministic plan suffix."""

    def __init__(self, draws: Sequence[SampleDraw]) -> None:
        self.draws = list(draws)

    def __iter__(self) -> Iterator[SampleDraw]:
        return iter(self.draws)

    def __len__(self) -> int:
        return len(self.draws)


class PaperDataset:
    def __init__(self, dataset_root: Path, rows: Sequence[dict[str, str]],
                 resolution: int, training: bool) -> None:
        self.root = Path(dataset_root).resolve()
        self.rows = list(rows)
        self.resolution = int(resolution)
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: int | SampleDraw) -> dict[str, Any]:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("training dataset requires torch") from error
        if isinstance(key, SampleDraw):
            index, augmentation_seed = key.row_index, key.augmentation_seed
        else:
            index = int(key)
            augmentation_seed = _seed(index, "validation")
        row = self.rows[index]
        with Image.open(self.root / row["image_path"]) as handle:
            image = handle.convert("RGB")
        with Image.open(self.root / row["mask_path"]) as handle:
            mask = handle.convert("L")
        if image.size != (self.resolution, self.resolution):
            image = image.resize(
                (self.resolution, self.resolution), Image.Resampling.LANCZOS)
            mask = mask.resize(
                (self.resolution, self.resolution), Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.uint8)
        mask_array = np.asarray(mask, dtype=np.uint8) >= 128
        if self.training:
            image_array, mask_array = self._augment(
                image_array, mask_array, augmentation_seed)
        image_float = image_array.astype(np.float32) / 255.0
        image_float = (image_float - IMAGENET_MEAN) / IMAGENET_STD
        distance = signed_distance_target(mask_array, radius_px=24.0)
        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(image_float.transpose(2, 0, 1))),
            "mask": torch.from_numpy(
                np.ascontiguousarray(mask_array[None]).astype(np.float32)),
            "distance": torch.from_numpy(
                np.ascontiguousarray(distance[None])),
            "sample_id": row["sample_id"],
            "capture_profile": row["capture_profile"],
            "sample_kind": row["sample_kind"],
        }

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray, seed: int
                 ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        rotations = int(rng.integers(0, 4))
        if rotations:
            image = np.rot90(image, rotations)
            mask = np.rot90(mask, rotations)
        if rng.random() < 0.5:
            image, mask = np.fliplr(image), np.fliplr(mask)
        if rng.random() < 0.5:
            image, mask = np.flipud(image), np.flipud(mask)
        value = image.astype(np.float32) / 255.0
        contrast = float(rng.uniform(0.90, 1.10))
        brightness = float(rng.uniform(-0.05, 0.05))
        gamma = float(rng.uniform(0.90, 1.10))
        value = np.clip((value - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
        value = np.power(value, gamma)
        if rng.random() < 0.25:
            value += rng.normal(0.0, rng.uniform(0.002, 0.012), value.shape)
        result = np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(result), np.ascontiguousarray(mask)


def load_rows(dataset_root: Path) -> list[dict[str, str]]:
    return read_sample_rows(dataset_root)
