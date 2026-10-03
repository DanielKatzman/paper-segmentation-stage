"""Mask diagnostics that are recorded before any corrective postprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from universal_segmenter.contracts import ReasonCode


@dataclass(frozen=True, slots=True)
class StructuralDiagnostics:
    area_fraction: float
    component_count: int
    second_component_fraction: float
    hole_fraction: float
    touches_border: bool


def diagnose_mask(mask: np.ndarray) -> StructuralDiagnostics:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    structure = np.ones((3, 3), dtype=bool)
    labels, count = ndimage.label(mask, structure=structure)
    areas = np.bincount(labels.ravel()) if count else np.zeros(1, dtype=int)
    foreground_areas = sorted((int(value) for value in areas[1:]), reverse=True)
    second = foreground_areas[1] if len(foreground_areas) > 1 else 0
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    touches = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    total = mask.size
    return StructuralDiagnostics(
        area_fraction=float(mask.mean()),
        component_count=int(count),
        second_component_fraction=float(second / total),
        hole_fraction=float(holes.mean()),
        touches_border=touches,
    )


def structural_reasons(diagnostics: StructuralDiagnostics,
                       minimum_area: float = 0.0004,
                       maximum_area: float = 0.94,
                       maximum_second_component: float = 0.0025,
                       maximum_hole_fraction: float = 0.0025) -> list[ReasonCode]:
    reasons: list[ReasonCode] = []
    if diagnostics.area_fraction < minimum_area:
        reasons.append(ReasonCode.NO_FRAGMENT)
    elif diagnostics.area_fraction > maximum_area:
        reasons.append(ReasonCode.IMPLAUSIBLE_MASK)
    if diagnostics.second_component_fraction > maximum_second_component:
        reasons.append(ReasonCode.MULTIPLE_FRAGMENTS)
    if diagnostics.hole_fraction > maximum_hole_fraction:
        reasons.append(ReasonCode.IMPLAUSIBLE_MASK)
    if diagnostics.touches_border:
        reasons.append(ReasonCode.FRAGMENT_CLIPPED)
    return list(dict.fromkeys(reasons))

