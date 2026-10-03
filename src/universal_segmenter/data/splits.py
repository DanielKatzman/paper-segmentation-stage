"""Connected-group split assignment that prevents page and session leakage."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any, Iterable


SPLITS = ("train", "validation", "calibration", "holdout")


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _tokens(record: dict[str, Any]) -> list[str]:
    explicit_group = str(record.get("split_group_id", "")).strip()
    if explicit_group:
        return [f"physical_piece:{explicit_group}"]
    tokens = [
        f"page:{record['document_id']}::{record['page_id']}",
        f"session:{record['capture_session_id']}",
    ]
    asset = str(record.get("background_asset_id", "")).strip()
    if asset:
        tokens.append(f"asset:{asset}")
    return tokens


def connected_capture_groups(records: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    records = list(records)
    union = _UnionFind()
    for record in records:
        tokens = _tokens(record)
        for token in tokens[1:]:
            union.union(tokens[0], token)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[union.find(_tokens(record)[0])].append(record)
    return [sorted(group, key=lambda row: row["capture_id"])
            for _, group in sorted(grouped.items())]


def _group_key(group: list[dict[str, Any]]) -> str:
    ids = "\n".join(sorted(str(record["capture_id"]) for record in group))
    return hashlib.sha256(ids.encode("utf-8")).hexdigest()[:16]


def assign_grouped_splits(
    records: Iterable[dict[str, Any]],
    ratios: dict[str, float] | None = None,
    seed: int = 64319,
) -> list[dict[str, str]]:
    records = list(records)
    ratios = ratios or {"train": 0.70, "validation": 0.15, "calibration": 0.15}
    if set(ratios) - set(SPLITS):
        raise ValueError(f"unknown split names: {set(ratios) - set(SPLITS)}")
    if abs(sum(ratios.values()) - 1.0) > 1.0e-8:
        raise ValueError("split ratios must sum to one")
    groups = connected_capture_groups(records)
    fixed: list[tuple[list[dict[str, Any]], str]] = []
    free: list[list[dict[str, Any]]] = []
    for group in groups:
        reserved = {str(row.get("reserved_split", "")).strip()
                    for row in group if str(row.get("reserved_split", "")).strip()}
        if len(reserved) > 1:
            raise ValueError(f"connected group has conflicting reserved splits: {reserved}")
        if reserved:
            split = next(iter(reserved))
            if split not in SPLITS:
                raise ValueError(f"invalid reserved_split: {split}")
            fixed.append((group, split))
        else:
            free.append(group)
    random.Random(seed).shuffle(free)
    free.sort(key=len, reverse=True)
    free_count = sum(len(group) for group in free)
    targets = {name: ratio * free_count for name, ratio in ratios.items()}
    counts = {name: 0 for name in ratios}
    grouped_assignments: list[tuple[list[dict[str, Any]], str, bool]] = []
    for group, split in fixed:
        grouped_assignments.append((group, split, True))
    for group in free:
        split = max(ratios, key=lambda name: (targets[name] - counts[name], -counts[name]))
        grouped_assignments.append((group, split, False))
        counts[split] += len(group)

    # A deficit-only greedy assignment can leave a small-ratio split empty when
    # the source consists of a few large leakage groups. Repair that condition
    # by moving the free group that least increases squared target error. This
    # preserves every page/session connection and remains deterministic.
    required = [name for name, ratio in ratios.items() if ratio > 0.0]
    if len(groups) < len(required):
        raise ValueError(
            f"cannot populate {len(required)} nonempty splits from only "
            f"{len(groups)} connected capture groups")

    def total_group_counts() -> dict[str, int]:
        result = {name: 0 for name in SPLITS}
        for group, split, _ in grouped_assignments:
            result[split] += len(group)
        return result

    def free_group_counts() -> dict[str, int]:
        result = {name: 0 for name in ratios}
        for _, split, fixed_assignment in grouped_assignments:
            if not fixed_assignment and split in result:
                result[split] += 1
        return result

    for empty_split in required:
        if total_group_counts()[empty_split] > 0:
            continue
        current_counts = total_group_counts()
        donor_group_counts = free_group_counts()
        candidates: list[tuple[float, str, str, int]] = []
        for index, (group, donor, fixed_assignment) in enumerate(grouped_assignments):
            if fixed_assignment or donor == empty_split:
                continue
            if donor in required and current_counts[donor] == len(group):
                continue
            if donor_group_counts.get(donor, 0) <= 1 and donor in required:
                continue
            trial = dict(counts)
            trial[donor] -= len(group)
            trial[empty_split] += len(group)
            error = sum((trial[name] - targets[name]) ** 2 for name in ratios)
            candidates.append((error, _group_key(group), donor, index))
        if not candidates:
            raise ValueError(
                f"could not populate required split {empty_split!r} without "
                "breaking a reserved or connected group")
        _, _, donor, index = min(candidates)
        group, _, _ = grouped_assignments[index]
        grouped_assignments[index] = (group, empty_split, False)
        counts[donor] -= len(group)
        counts[empty_split] += len(group)

    assignments = []
    for group, split, _ in grouped_assignments:
        key = _group_key(group)
        for record in group:
            assignments.append({
                "capture_id": str(record["capture_id"]),
                "group_id": key,
                "split": split,
            })
    return sorted(assignments, key=lambda row: row["capture_id"])


def extend_grouped_splits(
    records: Iterable[dict[str, Any]],
    existing_assignments: Iterable[dict[str, str]],
    ratios: dict[str, float] | None = None,
    seed: int = 64319,
) -> list[dict[str, str]]:
    """Assign only new connected groups while freezing every old split.

    A new capture that is connected to an existing page/session/background
    group inherits that group's split. If new metadata would connect captures
    that were previously placed in different splits, the extension is refused
    instead of silently introducing leakage or moving old evaluation data.
    """
    records = list(records)
    existing = list(existing_assignments)
    ratios = ratios or {
        "train": 0.70,
        "validation": 0.10,
        "calibration": 0.10,
        "holdout": 0.10,
    }
    if set(ratios) - set(SPLITS):
        raise ValueError(f"unknown split names: {set(ratios) - set(SPLITS)}")
    if abs(sum(ratios.values()) - 1.0) > 1.0e-8:
        raise ValueError("split ratios must sum to one")

    record_ids = [str(row["capture_id"]) for row in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("capture records contain duplicate capture IDs")
    by_existing: dict[str, dict[str, str]] = {}
    for row in existing:
        capture_id = str(row.get("capture_id", ""))
        split = str(row.get("split", ""))
        if not capture_id or split not in SPLITS:
            raise ValueError(f"invalid existing split assignment: {row}")
        if capture_id in by_existing:
            raise ValueError(f"duplicate existing split assignment: {capture_id}")
        by_existing[capture_id] = row
    missing_old = sorted(set(by_existing) - set(record_ids))
    if missing_old:
        raise ValueError(
            "new capture manifest is missing existing split members; first IDs: "
            f"{missing_old[:8]}")

    fixed_groups: list[tuple[list[dict[str, Any]], str]] = []
    free_groups: list[list[dict[str, Any]]] = []
    for group in connected_capture_groups(records):
        old_splits = {
            str(by_existing[str(row["capture_id"])]["split"])
            for row in group if str(row["capture_id"]) in by_existing
        }
        reserved = {
            str(row.get("reserved_split", "")).strip()
            for row in group if str(row.get("reserved_split", "")).strip()
        }
        invalid_reserved = reserved - set(SPLITS)
        if invalid_reserved:
            raise ValueError(f"invalid reserved_split values: {invalid_reserved}")
        required = old_splits | reserved
        if len(required) > 1:
            capture_ids = [str(row["capture_id"]) for row in group]
            raise ValueError(
                "new metadata connects captures assigned to conflicting fixed "
                f"splits {sorted(required)}: {capture_ids[:8]}")
        if required:
            fixed_groups.append((group, next(iter(required))))
        else:
            free_groups.append(group)

    random.Random(seed).shuffle(free_groups)
    free_groups.sort(key=len, reverse=True)
    targets = {
        name: float(ratio) * len(records) for name, ratio in ratios.items()
    }
    counts = {name: 0 for name in ratios}
    grouped: list[tuple[list[dict[str, Any]], str, bool]] = []
    for group, split in fixed_groups:
        grouped.append((group, split, True))
        if split in counts:
            counts[split] += len(group)
    for group in free_groups:
        split = max(
            ratios,
            key=lambda name: (targets[name] - counts[name], -counts[name], name),
        )
        grouped.append((group, split, False))
        counts[split] += len(group)

    required_splits = [name for name, ratio in ratios.items() if ratio > 0.0]
    for empty_split in required_splits:
        if counts[empty_split] > 0:
            continue
        candidates: list[tuple[float, str, int]] = []
        for index, (group, donor, fixed) in enumerate(grouped):
            if fixed or donor == empty_split:
                continue
            if donor in required_splits and counts[donor] == len(group):
                continue
            trial = dict(counts)
            trial[donor] -= len(group)
            trial[empty_split] += len(group)
            error = sum(
                (trial[name] - targets[name]) ** 2 for name in ratios)
            candidates.append((error, _group_key(group), index))
        if not candidates:
            raise ValueError(
                f"could not populate required split {empty_split!r} without "
                "moving an existing assignment")
        _, _, index = min(candidates)
        group, donor, _ = grouped[index]
        grouped[index] = (group, empty_split, False)
        counts[donor] -= len(group)
        counts[empty_split] += len(group)

    result: list[dict[str, str]] = []
    for group, split, _ in grouped:
        group_id = _group_key(group)
        for record in group:
            capture_id = str(record["capture_id"])
            result.append({
                "capture_id": capture_id,
                "group_id": group_id,
                "split": split,
            })
    result.sort(key=lambda row: row["capture_id"])
    changed = [
        row["capture_id"] for row in result
        if row["capture_id"] in by_existing
        and row["split"] != by_existing[row["capture_id"]]["split"]
    ]
    if changed:
        raise AssertionError(f"existing split assignments changed: {changed[:8]}")
    return result


def required_split_errors(
        assignments: Iterable[dict[str, str]],
        required: Iterable[str] = SPLITS,
        ) -> list[str]:
    """Report required splits that contain no assigned captures."""
    counts = defaultdict(int)
    for row in assignments:
        counts[str(row.get("split", ""))] += 1
    return [f"required split {name!r} has no captures"
            for name in required if counts[name] == 0]


def leakage_errors(records: Iterable[dict[str, Any]],
                   assignments: Iterable[dict[str, str]]) -> list[str]:
    by_capture = {row["capture_id"]: row["split"] for row in assignments}
    errors: list[str] = []
    for group in connected_capture_groups(records):
        splits = {by_capture.get(record["capture_id"], "MISSING") for record in group}
        if len(splits) != 1:
            captures = sorted(record["capture_id"] for record in group)
            errors.append(f"connected group crosses splits {sorted(splits)}: {captures}")
    return errors
