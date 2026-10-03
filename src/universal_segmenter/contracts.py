"""Versioned public data contracts for the standalone segmenter."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


RESULT_SCHEMA_VERSION = "universal-paper-segmentation-result-1.0"


class ReasonCode(str, Enum):
    NO_FRAGMENT = "NO_FRAGMENT"
    MULTIPLE_FRAGMENTS = "MULTIPLE_FRAGMENTS"
    FRAGMENT_CLIPPED = "FRAGMENT_CLIPPED"
    FRAGMENT_TOO_SMALL = "FRAGMENT_TOO_SMALL"
    LOW_BOUNDARY_CONFIDENCE = "LOW_BOUNDARY_CONFIDENCE"
    COARSE_FINE_DISAGREEMENT = "COARSE_FINE_DISAGREEMENT"
    CAPTURE_TOO_BLURRED = "CAPTURE_TOO_BLURRED"
    EXCESSIVE_GLARE = "EXCESSIVE_GLARE"
    IMPLAUSIBLE_MASK = "IMPLAUSIBLE_MASK"
    OUTSIDE_SUPPORTED_ENVELOPE = "OUTSIDE_SUPPORTED_ENVELOPE"

    def __str__(self) -> str:
        return self.value


@dataclass(slots=True)
class SegmentationResult:
    """One segmentation result in the original capture coordinate system."""

    mask: np.ndarray
    probability_map: np.ndarray | None
    contour_xy: list[tuple[float, float]]
    bbox_xywh: tuple[int, int, int, int]
    quality_score: float
    accepted: bool
    reason_codes: list[ReasonCode] = field(default_factory=list)
    diagnostics: dict[str, float | int | bool | str] = field(default_factory=dict)
    transform: dict[str, Any] = field(default_factory=dict)
    model_version: str = "unversioned"
    timings_ms: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.mask.ndim != 2 or self.mask.dtype != np.bool_:
            raise ValueError("mask must be a two-dimensional NumPy bool array")
        if self.probability_map is not None:
            if self.probability_map.shape != self.mask.shape:
                raise ValueError("probability_map must match mask dimensions")
            if not np.isfinite(self.probability_map).all():
                raise ValueError("probability_map contains non-finite values")
            minimum = float(self.probability_map.min(initial=0.0))
            maximum = float(self.probability_map.max(initial=1.0))
            if minimum < 0.0 or maximum > 1.0:
                raise ValueError("probability_map values must be in [0, 1]")
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be in [0, 1]")
        x, y, width, height = self.bbox_xywh
        if min(x, y, width, height) < 0 or width == 0 or height == 0:
            raise ValueError("bbox_xywh must be a positive in-image rectangle")
        if x + width > self.mask.shape[1] or y + height > self.mask.shape[0]:
            raise ValueError("bbox_xywh extends outside the mask")
        if self.accepted and self.reason_codes:
            raise ValueError("accepted results cannot contain rejection reasons")

    def metadata(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "image_size_hw": [int(self.mask.shape[0]), int(self.mask.shape[1])],
            "contour_xy": [[float(x), float(y)] for x, y in self.contour_xy],
            "bbox_xywh": [int(value) for value in self.bbox_xywh],
            "quality_score": float(self.quality_score),
            "accepted": bool(self.accepted),
            "reason_codes": [str(value) for value in self.reason_codes],
            "diagnostics": self.diagnostics,
            "transform": self.transform,
            "model_version": self.model_version,
            "timings_ms": self.timings_ms,
        }
