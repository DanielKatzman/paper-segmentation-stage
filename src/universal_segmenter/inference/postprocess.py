"""Small, explicit postprocessing policies calibrated without touching holdout."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


POSTPROCESS_VERSION = "paper-target-component-1.0"
VALID_POLICIES = ("raw", "largest_non_border")


def apply_postprocess(mask: np.ndarray, policy: str) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if policy == "raw":
        return value.copy()
    if policy != "largest_non_border":
        raise ValueError(f"unknown postprocess policy: {policy!r}")

    labels, count = ndimage.label(value, structure=np.ones((3, 3), dtype=bool))
    if count <= 1:
        return ndimage.binary_fill_holes(value).astype(bool)
    areas = np.bincount(labels.ravel())
    candidates: list[tuple[bool, int, int]] = []
    for label in range(1, count + 1):
        component = labels == label
        touches = bool(
            component[0].any() or component[-1].any()
            or component[:, 0].any() or component[:, -1].any())
        candidates.append((not touches, int(areas[label]), label))
    # Prefer a complete in-frame sheet. Within that class, keep the largest
    # component. If every component touches the border, keep the largest one.
    _, _, selected = max(candidates)
    return ndimage.binary_fill_holes(labels == selected).astype(bool)

