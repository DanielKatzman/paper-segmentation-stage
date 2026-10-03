"""Persistent, physical-piece-safe split assignment for annotation workspaces."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable

from universal_segmenter.data.physical_identity import (
    assign_physical_piece_identity,
)


SPLIT_NAMES = ("train", "validation", "calibration", "holdout")
SPLIT_LABELS = {
    "train": "Train",
    "validation": "Validation",
    "calibration": "Calibration",
    "holdout": "Final test",
}
DEFAULT_FOLDER_RATIOS = {
    "train": 0.70,
    "validation": 0.10,
    "calibration": 0.10,
    "holdout": 0.10,
}


def folder_key(record: dict[str, Any]) -> str:
    return str(
        record.get("page_folder")
        or record.get("capture_folder")
        or record.get("page_id")
        or "captures"
    ).replace("\\", "/")


def normalize_ratios(value: dict[str, float] | None) -> dict[str, float]:
    ratios = dict(DEFAULT_FOLDER_RATIOS if value is None else value)
    unknown = set(ratios) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"unknown dataset split names: {sorted(unknown)}")
    normalized = {name: float(ratios.get(name, 0.0)) for name in SPLIT_NAMES}
    if any(value < 0.0 for value in normalized.values()):
        raise ValueError("dataset split percentages cannot be negative")
    total = sum(normalized.values())
    if total <= 0.0:
        raise ValueError("at least one dataset split percentage must be positive")
    return {name: value / total for name, value in normalized.items()}


def normalize_folder_policies(
        folders: Iterable[str],
        policies: dict[str, dict[str, float]] | None,
        ) -> dict[str, dict[str, float]]:
    supplied = policies or {}
    return {
        folder: normalize_ratios(supplied.get(folder))
        for folder in sorted(set(folders))
    }


def _order_key(seed: int, folder: str, group_id: str) -> str:
    value = f"{seed}\n{folder}\n{group_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def assign_workspace_splits(
        records: list[dict[str, Any]],
        folder_policies: dict[str, dict[str, float]] | None = None,
        seed: int = 73191,
        reassign_automatic: bool = False,
        ) -> dict[str, Any]:
    """Assign every physical piece while preserving locked/manual choices.

    Percentages are balanced independently inside each page folder. A complete
    physical-piece group is the indivisible assignment unit, so the resulting
    capture percentages can differ slightly from the requested targets.
    """
    identity_method = assign_physical_piece_identity(records)
    folders = {folder_key(record) for record in records}
    policies = normalize_folder_policies(folders, folder_policies)
    by_folder: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        by_folder.setdefault(folder_key(record), {}).setdefault(
            str(record["split_group_id"]), []).append(record)

    for folder, groups_by_id in sorted(by_folder.items()):
        ratios = policies[folder]
        groups = list(groups_by_id.items())
        total_captures = sum(len(group) for _, group in groups)
        targets = {
            split: ratios[split] * total_captures for split in SPLIT_NAMES}
        counts = {split: 0 for split in SPLIT_NAMES}
        free: list[tuple[str, list[dict[str, Any]]]] = []

        for group_id, group in groups:
            existing = {
                str(record.get("dataset_split", "")).strip()
                for record in group
                if str(record.get("dataset_split", "")).strip()
                and (
                    not reassign_automatic
                    or record.get("split_assignment_source") == "manual"
                )
            }
            invalid = existing - set(SPLIT_NAMES)
            if invalid:
                raise ValueError(
                    f"physical piece {group_id} has invalid split values: "
                    f"{sorted(invalid)}")
            if len(existing) > 1:
                raise ValueError(
                    f"physical piece {group_id} crosses fixed dataset splits: "
                    f"{sorted(existing)}")
            if existing:
                split = next(iter(existing))
                for record in group:
                    record["dataset_split"] = split
                    record.setdefault("split_assignment_source", "automatic")
                counts[split] += len(group)
            else:
                free.append((group_id, group))

        free.sort(key=lambda item: _order_key(seed, folder, item[0]))
        free.sort(key=lambda item: len(item[1]), reverse=True)
        assigned_free: list[tuple[str, list[dict[str, Any]], str]] = []
        for group_id, group in free:
            eligible = [split for split in SPLIT_NAMES if ratios[split] > 0.0]
            split = max(
                eligible,
                key=lambda name: (
                    targets[name] - counts[name], -counts[name], name),
            )
            for record in group:
                record["dataset_split"] = split
                record["split_assignment_source"] = "automatic"
            counts[split] += len(group)
            assigned_free.append((group_id, group, split))

        # When a folder contains enough physical pieces, do not leave a
        # requested evaluation split empty merely because train has the largest
        # numerical deficit. Move only newly assigned automatic groups; fixed
        # and manual choices are never changed by this repair.
        required = [split for split in SPLIT_NAMES if ratios[split] > 0.0]
        if len(groups) >= len(required):
            for empty in required:
                if counts[empty] > 0:
                    continue
                group_counts = Counter(
                    str(group[0]["dataset_split"])
                    for _, group in groups)
                candidates: list[tuple[float, str, int]] = []
                for index, (group_id, group, donor) in enumerate(assigned_free):
                    if donor == empty or group_counts[donor] <= 1:
                        continue
                    trial = dict(counts)
                    trial[donor] -= len(group)
                    trial[empty] += len(group)
                    error = sum(
                        (trial[split] - targets[split]) ** 2
                        for split in SPLIT_NAMES)
                    candidates.append((error, group_id, index))
                if not candidates:
                    continue
                _, _, index = min(candidates)
                group_id, group, donor = assigned_free[index]
                for record in group:
                    record["dataset_split"] = empty
                assigned_free[index] = (group_id, group, empty)
                counts[donor] -= len(group)
                counts[empty] += len(group)

    for record in records:
        record["split_assignment_folder"] = folder_key(record)
        record["split_assignment_seed"] = int(seed)

    return {
        "physical_identity_method": identity_method,
        "folders": policies,
        "captures_by_split": dict(Counter(
            str(record["dataset_split"]) for record in records)),
        "physical_pieces": len({
            str(record["split_group_id"]) for record in records}),
    }


def set_group_split(records: list[dict[str, Any]], group_id: str,
                    split: str) -> int:
    if split not in SPLIT_NAMES:
        raise ValueError(f"invalid dataset split: {split!r}")
    changed = 0
    for record in records:
        if str(record.get("split_group_id", "")) != group_id:
            continue
        changed += int(record.get("dataset_split") != split)
        record["dataset_split"] = split
        record["split_assignment_source"] = "manual"
    if not any(str(record.get("split_group_id", "")) == group_id
               for record in records):
        raise ValueError(f"unknown physical-piece split group: {group_id}")
    return changed


def split_manifest_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    output = []
    for record in records:
        split = str(record.get("dataset_split", ""))
        if split not in SPLIT_NAMES:
            raise ValueError(
                f"capture {record.get('capture_id')} has no valid dataset split")
        output.append({
            "schema_version": "universal-paper-split-manifest-1.0",
            "capture_id": str(record["capture_id"]),
            "group_id": str(record["split_group_id"]),
            "split": split,
        })
    return sorted(output, key=lambda row: row["capture_id"])
