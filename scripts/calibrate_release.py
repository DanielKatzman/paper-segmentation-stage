"""Select threshold/postprocessing using calibration only, never holdout."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from universal_segmenter.evaluation import calibrate
from universal_segmenter.inference.candidates import final_model_spec


def main() -> None:
    spec = final_model_spec(PROJECT)
    export_metadata = json.loads(
        spec.onnx_path.with_suffix(".json").read_text(encoding="utf-8"))
    report = calibrate(
        PROJECT / "data" / "complete_dataset_v3",
        spec.onnx_path,
        export_metadata,
        PROJECT / "exports" / "calibration.json",
        cpu_threads=7,
    )
    print(json.dumps(report, indent=2))
    print("\nCALIBRATION LOCKED. Do not change it after opening holdout.")


if __name__ == "__main__":
    main()

