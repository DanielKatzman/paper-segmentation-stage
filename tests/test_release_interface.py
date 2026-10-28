import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from universal_segmenter.pipeline import runner
from universal_segmenter import release_cli
from universal_segmenter.contracts import SegmentationResult


class FakeSegmenter:
    calls = 0
    def __init__(self, **kwargs):
        self.release = SimpleNamespace(model_version='test', model_sha256='a'*64,
            cache_namespace='test', threshold=0.5, postprocess_policy='raw', dataset_fingerprint='fixture')

    def segment(self, rgb):
        type(self).calls += 1
        mask = np.zeros(rgb.shape[:2], dtype=bool); mask[8:24, 10:30] = True
        return SegmentationResult(mask=mask, probability_map=mask.astype(np.float32),
                                  contour_xy=[], bbox_xywh=(10, 8, 20, 16), quality_score=1,
                                  accepted=True, model_version='test')


def test_headless_all_requires_review_and_never_opens_editor(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'Segmenter', FakeSegmenter)
    monkeypatch.setattr(runner, 'MaskEditor', lambda *a, **k: (_ for _ in ()).throw(AssertionError('GUI opened')))
    captures = tmp_path/'captures'; captures.mkdir()
    Image.new('RGB', (40, 32), 'gray').save(captures/'p.png')
    output = tmp_path/'output'
    report = runner.run_pipeline(captures, output, review='all', headless=True)
    row = json.loads((output/'segmentation_manifest.jsonl').read_text())
    assert report['needs_review'] == 1
    assert not row['accepted'] and row['decision_source'] == 'automatic'
    assert row['review_status'] == 'needs_review'
    assert release_cli.verify_output(output)['status'] == 'review_required'
    assert runner.run_pipeline(captures, output, review='all', headless=True)['needs_review'] == 1


def test_v2_resume_rejects_tampering_and_changed_request(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'Segmenter', FakeSegmenter)
    monkeypatch.setattr(release_cli, 'verify_models', lambda p: SimpleNamespace(
        model_sha256='a'*64, release_path=p/'release.json'))
    models = tmp_path/'models'; models.mkdir(); (models/'release.json').write_text('{}')
    captures = tmp_path/'captures'; captures.mkdir()
    Image.new('RGB', (40, 32), 'gray').save(captures/'p.png')
    output = tmp_path/'output'
    args = ['run', '--input', str(captures), '--output', str(output), '--models', str(models), '--review', 'none']
    assert release_cli.main(args) == 0
    assert release_cli.main(args+['--resume']) == 0
    assert release_cli.main(args+['--resume', '--pdf-dpi', '200']) == 2
    (output/'piece_masks'/next((output/'piece_masks').iterdir()).name).write_bytes(b'corrupted')
    assert release_cli.main(args+['--resume']) == 2
    assert not (output/'.paper-stage1.lock').exists()


def test_output_inspection_rejects_escaping_path(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'Segmenter', FakeSegmenter)
    source = tmp_path/'captures'; source.mkdir()
    Image.new('RGB', (40, 32), 'gray').save(source/'p.png')
    output = tmp_path/'output'; runner.run_pipeline(source, output, review='none')
    path = output/'segmentation_manifest.jsonl'; row = json.loads(path.read_text())
    row['piece_path'] = '../outside.png'; path.write_text(json.dumps(row)+'\n')
    import pytest
    with pytest.raises(ValueError, match='artifact path'):
        release_cli.verify_output(output)


def test_partial_resume_recomputes_a_corrupted_prediction_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'Segmenter', FakeSegmenter)
    source = tmp_path/'captures'; source.mkdir()
    Image.new('RGB', (40, 32), 'gray').save(source/'p.png')
    output = tmp_path/'output'
    FakeSegmenter.calls = 0
    runner.run_pipeline(source, output, review='none')
    cache = next((output/'cache').rglob('*.png'))
    Image.new('L', (40, 32), 0).save(cache)
    runner.run_pipeline(source, output, review='none', reprocess=True)
    assert FakeSegmenter.calls == 2
    row = json.loads((output/'segmentation_manifest.jsonl').read_text())
    assert row['accepted']
