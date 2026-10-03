"""Export and parity-check the exact completed v3 best checkpoint."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from universal_segmenter.inference.candidates import final_model_spec
from universal_segmenter.inference.onnx_export import ensure_export


EXPECTED_DATASET_FINGERPRINT = (
    "225061bdeab36f7a10d5b9080c01ad013db0cb3b798d53e6ddfddcd86834ffaf")
EXPECTED_BEST_EPOCH_ONE_BASED = 8
EXPECTED_BEST_SCORE = 0.9259111612330179


def main() -> None:
    metadata = ensure_export(final_model_spec(PROJECT), cpu_threads=7, force=True)
    if metadata["dataset_fingerprint"] != EXPECTED_DATASET_FINGERPRINT:
        raise RuntimeError("best.pt belongs to a different frozen dataset")
    if int(metadata["best_epoch_one_based"]) != EXPECTED_BEST_EPOCH_ONE_BASED:
        raise RuntimeError("best.pt is not the approved human epoch 8 checkpoint")
    if not math.isclose(
            float(metadata["best_validation_score"]), EXPECTED_BEST_SCORE,
            rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError("best.pt has an unexpected validation score")
    print(json.dumps(metadata, indent=2))
    print("\nEXPORT VERIFIED:", final_model_spec(PROJECT).onnx_path)


if __name__ == "__main__":
    main()

