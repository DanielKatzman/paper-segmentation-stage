"""Per-image binary region metrics."""

from __future__ import annotations

import numpy as np


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must be same-shaped 2D masks")
    true_positive = int(np.count_nonzero(prediction & target))
    false_positive = int(np.count_nonzero(prediction & ~target))
    false_negative = int(np.count_nonzero(~prediction & target))
    union = true_positive + false_positive + false_negative
    predicted = true_positive + false_positive
    actual = true_positive + false_negative
    return {
        "iou": 1.0 if union == 0 else true_positive / union,
        "dice": 1.0 if predicted + actual == 0 else 2.0 * true_positive / (predicted + actual),
        "precision": 1.0 if predicted == 0 and actual == 0 else (
            0.0 if predicted == 0 else true_positive / predicted),
        "recall": 1.0 if actual == 0 else true_positive / actual,
        "false_positive_fraction": false_positive / prediction.size,
        "false_negative_fraction": false_negative / prediction.size,
    }

