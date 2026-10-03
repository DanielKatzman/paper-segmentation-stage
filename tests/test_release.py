import hashlib
import json

from universal_segmenter.inference.candidates import (
    FINAL_MODEL_RELEASE,
    MODEL_FILENAME,
    MODEL_KEY,
    RUN_NAME,
)
from universal_segmenter.inference.release import RELEASE_SCHEMA, load_release


def test_release_loader_verifies_model_identity(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    model = models / MODEL_FILENAME
    model.write_bytes(b"test-model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    (models / "release.json").write_text(json.dumps({
        "schema_version": RELEASE_SCHEMA,
        "model_release": FINAL_MODEL_RELEASE,
        "model_type": MODEL_KEY,
        "checkpoint_run_name": RUN_NAME,
        "resolution": 768,
        "onnx_sha256": digest,
        "dataset_fingerprint": "abc",
        "calibration": {
            "threshold": 0.55,
            "postprocess_policy": "largest_non_border",
        },
        "holdout_evaluation": {"valid": True},
    }), encoding="utf-8")
    release = load_release(tmp_path)
    assert release.threshold == 0.55
    assert release.postprocess_policy == "largest_non_border"
    assert release.model_sha256 == digest

