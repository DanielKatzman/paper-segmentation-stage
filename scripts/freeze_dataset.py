"""Validate and freeze the dataset before training or release evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from universal_segmenter.training import freeze_dataset


DATASET_ROOT = PROJECT / "data" / "complete_dataset_v3"


if __name__ == "__main__":
    report = freeze_dataset(DATASET_ROOT)
    print(json.dumps(report, indent=2))
    print("\nFROZEN:", DATASET_ROOT / "dataset_freeze.json")

