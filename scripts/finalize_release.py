"""Create immutable production model files only after every gate passes."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from universal_segmenter.evaluation.release_gate import CALIBRATION_SCHEMA, HOLDOUT_SCHEMA
from universal_segmenter.inference.candidates import FINAL_MODEL_RELEASE, MODEL_FILENAME, MODEL_KEY, RUN_NAME, final_model_spec
from universal_segmenter.inference.onnx_export import EXPORT_SCHEMA, sha256_file
from universal_segmenter.inference.release import RELEASE_SCHEMA, load_release


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    spec = final_model_spec(PROJECT)
    export = _read(spec.onnx_path.with_suffix(".json"))
    calibration = _read(PROJECT / "exports" / "calibration.json")
    holdout = _read(PROJECT / "exports" / "holdout_evaluation.json")
    if export.get("schema_version") != EXPORT_SCHEMA:
        raise RuntimeError("unexpected export metadata")
    if calibration.get("schema_version") != CALIBRATION_SCHEMA:
        raise RuntimeError("unexpected calibration metadata")
    if holdout.get("schema_version") != HOLDOUT_SCHEMA:
        raise RuntimeError("unexpected holdout metadata")
    if not holdout.get("valid"):
        raise RuntimeError(
            "holdout gates did not pass; do not package this checkpoint")
    hashes = {
        export.get("onnx_sha256"), calibration.get("onnx_sha256"),
        holdout.get("onnx_sha256")}
    fingerprints = {
        export.get("dataset_fingerprint"), calibration.get("dataset_fingerprint"),
        holdout.get("dataset_fingerprint")}
    if len(hashes) != 1 or len(fingerprints) != 1:
        raise RuntimeError("export/calibration/holdout identity mismatch")

    model_folder = PROJECT / "models"
    model_folder.mkdir(parents=True, exist_ok=True)
    production_model = model_folder / MODEL_FILENAME
    temporary_model = production_model.with_suffix(".onnx.tmp")
    shutil.copy2(spec.onnx_path, temporary_model)
    os.replace(temporary_model, production_model)
    if sha256_file(production_model) != export["onnx_sha256"]:
        raise RuntimeError("copied production ONNX failed checksum verification")
    release = {
        "schema_version": RELEASE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_release": FINAL_MODEL_RELEASE,
        "model_type": MODEL_KEY,
        "checkpoint_run_name": RUN_NAME,
        "resolution": int(export["resolution"]),
        "onnx_filename": MODEL_FILENAME,
        "onnx_sha256": export["onnx_sha256"],
        "dataset_fingerprint": export["dataset_fingerprint"],
        "best_epoch_one_based": export["best_epoch_one_based"],
        "best_validation_score": export["best_validation_score"],
        "parity": export["parity"],
        "calibration": calibration,
        "holdout_evaluation": holdout,
    }
    release_path = model_folder / "release.json"
    temporary = release_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, release_path)
    verified = load_release(PROJECT, require_holdout_pass=True)
    print(json.dumps({
        "model": str(verified.model_path),
        "release": str(verified.release_path),
        "sha256": verified.model_sha256,
        "threshold": verified.threshold,
        "postprocess_policy": verified.postprocess_policy,
    }, indent=2))


if __name__ == "__main__":
    main()

