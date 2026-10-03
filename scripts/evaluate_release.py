"""Run the one-time frozen holdout gate after calibration is locked."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from universal_segmenter.evaluation import evaluate_holdout
from universal_segmenter.inference.candidates import final_model_spec


def main() -> None:
    spec = final_model_spec(PROJECT)
    export_metadata = json.loads(
        spec.onnx_path.with_suffix(".json").read_text(encoding="utf-8"))
    calibration = json.loads(
        (PROJECT / "exports" / "calibration.json").read_text(encoding="utf-8"))
    report = evaluate_holdout(
        PROJECT / "data" / "complete_dataset_v3",
        spec.onnx_path,
        export_metadata,
        calibration,
        PROJECT / "exports" / "holdout_evaluation.json",
        PROJECT / "exports" / "holdout_preview.jpg",
        cpu_threads=7,
    )
    print(json.dumps(report, indent=2))
    print("\nHOLDOUT VALID:", report["valid"])
    print("Inspect exports/holdout_preview.jpg before finalizing.")


if __name__ == "__main__":
    main()

