from types import SimpleNamespace

import numpy as np
from PIL import Image

from universal_segmenter.contracts import SegmentationResult
from universal_segmenter.data.manifest import read_jsonl
from universal_segmenter.pipeline import runner


class FakeSegmenter:
    calls = 0

    def __init__(self, **_kwargs):
        self.release = SimpleNamespace(
            model_version="test-release",
            model_sha256="a" * 64,
            cache_namespace="test-release_aaaaaaaa_t500_raw",
            threshold=0.5,
            postprocess_policy="raw",
            dataset_fingerprint="dataset",
        )

    def segment(self, rgb):
        type(self).calls += 1
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[8:24, 10:30] = True
        return SegmentationResult(
            mask=mask,
            probability_map=mask.astype(np.float32),
            contour_xy=[],
            bbox_xywh=(10, 8, 20, 16),
            quality_score=1.0,
            accepted=True,
            model_version="test-release:aaaaaaaaaaaaaaaa",
        )


def test_batch_contract_is_resumable(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "Segmenter", FakeSegmenter)
    source = tmp_path / "captures"
    source.mkdir()
    Image.new("RGB", (40, 32), "gray").save(source / "piece.jpg")
    output = tmp_path / "output"
    first = runner.run_pipeline(source, output, review="none")
    second = runner.run_pipeline(source, output, review="none")
    assert first["complete"] is True
    assert second["reused_complete_outputs"] == 1
    assert FakeSegmenter.calls == 1
    row = read_jsonl(output / "segmentation_manifest.jsonl")[0]
    assert row["schema_version"] == runner.PIPELINE_SCHEMA
    assert row["accepted"] is True
    assert (output / row["piece_path"]).is_file()
