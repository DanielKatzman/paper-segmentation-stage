"""Boundary F1, ASSD, and robust Hausdorff metrics."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return mask ^ ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))


def boundary_metrics(prediction: np.ndarray, target: np.ndarray,
                     tolerance_px: float) -> dict[str, float]:
    predicted_boundary, target_boundary = boundary(prediction), boundary(target)
    predicted_count = int(predicted_boundary.sum())
    target_count = int(target_boundary.sum())
    if predicted_count == 0 and target_count == 0:
        return {"boundary_precision": 1.0, "boundary_recall": 1.0, "boundary_f1": 1.0}
    if predicted_count == 0 or target_count == 0:
        return {"boundary_precision": 0.0, "boundary_recall": 0.0, "boundary_f1": 0.0}
    target_distance = ndimage.distance_transform_edt(~target_boundary)
    predicted_distance = ndimage.distance_transform_edt(~predicted_boundary)
    precision = float(np.mean(target_distance[predicted_boundary] <= tolerance_px))
    recall = float(np.mean(predicted_distance[target_boundary] <= tolerance_px))
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
    }


def surface_distances(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted_boundary, target_boundary = boundary(prediction), boundary(target)
    if not predicted_boundary.any() and not target_boundary.any():
        return {"assd_px": 0.0, "hd95_px": 0.0}
    if not predicted_boundary.any() or not target_boundary.any():
        return {"assd_px": float("inf"), "hd95_px": float("inf")}
    target_distance = ndimage.distance_transform_edt(~target_boundary)[predicted_boundary]
    predicted_distance = ndimage.distance_transform_edt(~predicted_boundary)[target_boundary]
    distances = np.concatenate((target_distance, predicted_distance))
    return {
        "assd_px": float(distances.mean()),
        "hd95_px": float(np.percentile(distances, 95)),
    }

