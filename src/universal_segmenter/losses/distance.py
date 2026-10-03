"""Signed-distance supervision target."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def signed_distance_target(mask: np.ndarray, radius_px: float = 24.0) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if radius_px <= 0:
        raise ValueError("radius_px must be positive")
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    signed = inside - outside
    return np.clip(signed / radius_px, -1.0, 1.0).astype(np.float32)

