import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from universal_segmenter import release_cli
from universal_segmenter.pipeline import multi
from universal_segmenter.scenes.core import new_scene
from universal_segmenter.scenes.workspace import propose


class FakeSceneSegmenter:
    calls = 0
    count = 2

    def __init__(self,**kwargs):
        self.release = SimpleNamespace(model_version="fixture",model_sha256="a"*64,
            cache_namespace="fixture",threshold=.5,dataset_fingerprint="fixture")

    def segment_scene(self,rgb,minimum_pixels=4):
        type(self).calls += 1
        scene = new_scene(rgb)
        probability = np.zeros(rgb.shape[:2],np.float32)
        if self.count:
            probability[4:15,5:18]=.9
        if self.count>1:
            probability[21:34,25:38]=.9
        propose(scene,probability,.5,"fixture",minimum_pixels)
        return scene


@pytest.fixture
def source(tmp_path,monkeypatch):
    monkeypatch.setattr(multi,"Segmenter",FakeSceneSegmenter)
    monkeypatch.setattr(FakeSceneSegmenter,"count",2)
    monkeypatch.setattr(FakeSceneSegmenter,"calls",0)
    path=tmp_path/"inputs";path.mkdir()
    Image.new("RGB",(48,40),"gray").save(path/"photo.png")
    return path


def rows(output):
    return [json.loads(line) for line in (output/"segmentation_manifest.jsonl").read_text().splitlines()]


def test_release_defaults_to_one_native_entry_per_piece(source,tmp_path,monkeypatch):
    models=tmp_path/"models";models.mkdir();(models/"release.json").write_text("{}")
    monkeypatch.setattr(release_cli,"verify_models",lambda p:SimpleNamespace(
        model_sha256="a"*64,release_path=p/"release.json"))
    output=tmp_path/"output"
    args=["run","--input",str(source),"--output",str(output),"--models",str(models),"--review","none"]
    assert release_cli.main(args)==0
    values=rows(output)
    assert len(values)==2 and len({r["capture_id"] for r in values})==2
    assert len({r["source_capture_id"] for r in values})==1
    assert len({r["source_sha256"] for r in values})==1
    for key in ("mask_path","piece_path","piece_mask_path","overlay_path"):
        assert len({r[key] for r in values})==2
    for row in values:
        assert row["original_size_hw"]==[40,48]
        mask=np.asarray(Image.open(output/row["mask_path"]))
        x,y,w,h=row["bbox_xywh"]
        rgba=np.asarray(Image.open(output/row["piece_path"]))
        assert np.array_equal(rgba[:,:,3],mask[y:y+h,x:x+w])
    report=json.loads((output/"segmentation_run.json").read_text())
    assert report["source_count"]==1 and report["piece_count"]==2
    assert release_cli.main(args+["--resume"])==0
    assert FakeSceneSegmenter.calls==1
    assert release_cli.main(args+["--resume","--piece-mode","single"])==2


def test_each_proposal_requires_review_independently(source,tmp_path):
    output=tmp_path/"output"
    report=multi.run_multi_pipeline(source,output,review="all",headless=True)
    assert report["needs_review"]==2 and report["complete"]
    assert all(not r["accepted"] for r in rows(output))
    assert release_cli.verify_output(output)["status"]=="review_required"


def test_interactive_cancel_preserves_completed_piece_and_consistent_source_inventory(source,tmp_path,monkeypatch):
    from universal_segmenter.annotation import editor
    actions=iter(["save","quit"])
    monkeypatch.setattr(editor,"MaskEditor",lambda rgb,mask,*args,**kwargs:
        SimpleNamespace(run=lambda:(next(actions),mask.copy())))
    output=tmp_path/"output"
    report=multi.run_multi_pipeline(source,output,review="all",headless=False)
    assert report["stopped_safely"] and not report["complete"]
    assert len(rows(output))==1 and rows(output)[0]["accepted"]
    assert not report["sources"][0]["processed"]
    assert release_cli.verify_output(output)["status"]=="review_required"


def test_source_mutation_during_inference_cannot_publish_receipt(source,tmp_path,monkeypatch):
    original=FakeSceneSegmenter.segment_scene
    def changed(self,rgb,minimum_pixels=4):
        scene=original(self,rgb,minimum_pixels)
        Image.new("RGB",(48,40),"red").save(source/"photo.png")
        return scene
    monkeypatch.setattr(FakeSceneSegmenter,"segment_scene",changed)
    models=tmp_path/"models";models.mkdir();(models/"release.json").write_text("{}")
    monkeypatch.setattr(release_cli,"verify_models",lambda p:SimpleNamespace(
        model_sha256="a"*64,release_path=p/"release.json"))
    output=tmp_path/"output"
    assert release_cli.main(["run","--input",str(source),"--output",str(output),
        "--models",str(models),"--review","none"])==2
    assert not (output/"integration_receipt.json").exists()
    assert not (output/".paper-stage1.lock").exists()


def test_empty_prediction_is_explicit_pending_observation(source,tmp_path,monkeypatch):
    monkeypatch.setattr(FakeSceneSegmenter,"count",0)
    output=tmp_path/"output"
    report=multi.run_multi_pipeline(source,output,review="none",headless=True)
    assert report["piece_count"]==0 and report["source_count"]==1
    assert len(rows(output))==1 and rows(output)[0]["empty_proposal"]
    assert not rows(output)[0]["accepted"]
    assert release_cli.verify_output(output)["status"]=="review_required"


def test_single_piece_preserves_original_capture_identity(source,tmp_path,monkeypatch):
    monkeypatch.setattr(FakeSceneSegmenter,"count",1)
    output=tmp_path/"output"
    multi.run_multi_pipeline(source,output,review="none",headless=True)
    row=rows(output)[0]
    assert row["capture_id"]==row["source_capture_id"]=="capture_"+row["source_sha256"]
    assert row["artifact_id"]==row["source_sha256"]


def test_corrupted_scene_cache_is_recomputed(source,tmp_path):
    output=tmp_path/"output"
    multi.run_multi_pipeline(source,output,review="none",headless=True)
    next((output/"cache").rglob("*.npz")).write_bytes(b"damaged")
    multi.run_multi_pipeline(source,output,review="none",headless=True,reprocess=True)
    assert FakeSceneSegmenter.calls==2
    assert release_cli.verify_output(output)["status"]=="ready"


def test_inspection_rejects_missing_observation_and_overlapping_pieces(source,tmp_path):
    output=tmp_path/"output"
    multi.run_multi_pipeline(source,output,review="none",headless=True)
    path=output/"segmentation_run.json"
    report=json.loads(path.read_text());original=path.read_text()
    report["sources"][0]["observation_ids"].pop()
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="inventory"):
        release_cli.verify_output(output)
    path.write_text(original)
    values=rows(output)
    for key in ("mask_path","piece_path","piece_mask_path","bbox_xywh"):
        values[1][key]=values[0][key]
    (output/"segmentation_manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in values))
    with pytest.raises(ValueError,match="overlap"):
        release_cli.verify_output(output)
