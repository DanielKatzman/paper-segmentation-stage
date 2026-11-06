import copy
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from universal_segmenter.scenes.core import SceneStore, new_scene, validate_scene, read_json, write_json
from universal_segmenter.scenes.workspace import (accept_instance, add_instance, edit_instance,
    finish_scene, merge_instances, propose, publish_pieces, reject_instance, split_instance)
from universal_segmenter.scenes.exchange import export_collection, import_collection
from universal_segmenter.scenes.dataset import assign_splits, build_dataset, load_view, verify_dataset, instance_metrics


def rgb(seed=1):
    return np.random.default_rng(seed).integers(40,180,(80,120,3),dtype=np.uint8)


def metadata(page="p1",session="s1",**kw):
    return {"contributor":"test","document_id":"doc1","page_id":page,
            "capture_session_id":session,"capture_profile":"scanner",**kw}


def mask(left=10):
    result = np.zeros((80,120),bool)
    result[10:40,left:left+20] = True
    return result


def completed(store,seed=1,**kw):
    scene = store.add(rgb(seed),metadata(**kw))
    add_instance(scene,mask(10),accepted=True)
    add_instance(scene,mask(60),accepted=True)
    finish_scene(scene)
    store.save(scene)
    return scene


def test_review_completeness_and_true_negatives(tmp_path):
    store = SceneStore(tmp_path)
    scene = store.add(rgb(),metadata())
    ident = add_instance(scene,mask())
    with pytest.raises(ValueError,match="Review every"):
        finish_scene(scene)
    reject_instance(scene,ident)
    assert not scene.eligible
    with pytest.raises(ValueError,match="not exhaustively"):
        validate_scene(scene,require_complete=True)
    finish_scene(scene,empty=True,clean_background=True)
    store.save(scene)
    assert store.load(scene.record["capture_id"]).eligible
    assert store.verify()["empty"]==1


def test_proposals_preserve_all_components_and_record_specks():
    scene = new_scene(rgb(),metadata())
    probability = (mask(10)|mask(60)).astype(np.float32)*.9
    probability[60,100]=.9
    propose(scene,probability,.3,"model")
    assert len(scene.active)==2
    assert len(scene.record["proposal"]["excluded_specks"])==1
    assert not scene.eligible
    with pytest.raises(ValueError,match="replace existing"):
        propose(scene,probability,.3,"model")


def test_unrelated_fragments_without_page_identity_feed_training_and_generator(tmp_path):
    from universal_segmenter.scenes.generator import collection_assets
    store = SceneStore(tmp_path/"work")
    scenes = []
    for count in (0,1,2):
        scene = store.add(rgb(40+count),{"capture_profile":"phone","split":"train"})
        for left in [10,60][:count]:
            add_instance(scene,mask(left),accepted=True)
        finish_scene(scene,empty=count==0,clean_background=count==0)
        store.save(scene)
        scenes.append(scene)
    export_collection(store,tmp_path/"arbitrary.zip")
    with zipfile.ZipFile(tmp_path/"arbitrary.zip") as archive:
        import io
        for scene in scenes:
            outline = np.asarray(Image.open(io.BytesIO(archive.read(
                "outlines/"+scene.record["capture_id"]+".png"))))
            assert outline.shape == scene.labels.shape
            assert not np.any((outline>0)&(scene.labels==0))
            assert bool(outline.any()) == bool(scene.active)
            if scene.active:
                assert outline[10,10] == 255 and outline[20,20] == 0
    imported = import_collection(tmp_path/"arbitrary.zip",tmp_path/"central")
    assert len(imported["accepted"]) == 3
    ledger = tmp_path/"ledger.json"
    assign_splits(scenes,ledger)
    inventory = tmp_path/"assets.json"
    write_json(inventory,{"schema_version":"paper-scene-source-inventory-1.0","assets":[]})
    assert collection_assets(store,inventory,ledger)["assets_added"] == 4  # Three cutouts and a clean background.
    assert build_dataset([store.root],tmp_path/"dataset",ledger)["sample_count"] == 3


def test_export_without_review_is_immediate_and_does_not_decode_sources(tmp_path,monkeypatch):
    store = SceneStore(tmp_path/"work")
    store.add(rgb(),{"capture_profile":"phone"})
    def unexpected(*args,**kwargs):
        raise AssertionError("Unreviewed captures must not be decoded during export")
    monkeypatch.setattr(store,"load",unexpected)
    monkeypatch.setattr(store,"rgb",unexpected)
    with pytest.raises(ValueError,match="No complete/confirmed-empty"):
        export_collection(store,tmp_path/"empty.zip")


def test_cropped_validation_detects_pixels_outside_bbox_and_large_label_values():
    scene = new_scene(rgb(),{"capture_profile":"phone"})
    add_instance(scene,mask(),accepted=True)
    scene.labels[scene.labels==1] = np.iinfo(np.int32).max
    scene.record["instances"][0]["label"] = int(np.iinfo(np.int32).max)
    validate_scene(scene)
    scene.labels[60,100] = np.iinfo(np.int32).max
    with pytest.raises(ValueError,match="bbox is stale"):
        validate_scene(scene)


def test_cropped_proposals_preserve_native_threshold_components():
    from scipy import ndimage
    probability = np.random.default_rng(731).random((80,120),dtype=np.float32)
    source,count = ndimage.label(probability>=.8,np.ones((3,3),bool))
    sizes = np.bincount(source.ravel())
    expected = {frozenset(np.flatnonzero(source==i)) for i in range(1,count+1) if sizes[i]>=4}
    scene = new_scene(rgb())
    propose(scene,probability,.8,"v3")
    actual = {frozenset(np.flatnonzero(scene.labels==i["label"])) for i in scene.active}
    assert actual == expected
    assert len({i["instance_id"] for i in scene.active}) == len(expected)
    validate_scene(scene)


def test_independent_edit_undo_and_stale_revision(tmp_path):
    store = SceneStore(tmp_path)
    scene = completed(store)
    old = scene.copy()
    other = scene.labels==scene.active[1]["label"]
    ident = scene.active[0]["instance_id"]
    edit_instance(scene,ident,mask(15),accept=True)
    store.save(scene)
    assert np.array_equal(scene.labels==scene.active[1]["label"],other)
    assert scene.active[0]["instance_id"]==ident
    with pytest.raises(ValueError,match="changed since"):
        store.save(old)
    restored = store.restore(scene.record["capture_id"],old.record["revision"])
    assert np.array_equal(restored.labels,old.labels)
    assert restored.record["parent_revision"]==scene.record["revision"]


def test_split_merge_is_explicit_and_overlaps_fail():
    scene = new_scene(rgb(),metadata())
    ident = add_instance(scene,mask(10)|mask(60))
    children = split_instance(scene,ident)
    assert len(children)==2
    with pytest.raises(ValueError,match="overlaps"):
        edit_instance(scene,children[0],mask(10)|mask(60))
    merged = merge_instances(scene,children)
    with pytest.raises(ValueError,match="disconnected"):
        accept_instance(scene,merged)
    assert instance_metrics(scene.labels,scene.labels)["exact_count"]


def test_approved_publication_has_unique_artifacts(tmp_path):
    store = SceneStore(tmp_path/"work")
    scene = completed(store)
    rejected = add_instance(scene,mask(90))
    reject_instance(scene,rejected)
    finish_scene(scene); store.save(scene)
    from PIL.PngImagePlugin import PngInfo
    from universal_segmenter.scenes.core import file_hash
    image_path=store.root/scene.record["image_path"]
    png_info=PngInfo();png_info.add_text("capture_context","source-byte-identity fixture")
    Image.fromarray(store.rgb(scene)).save(image_path,pnginfo=png_info)
    scene.record["image_sha256"]=file_hash(image_path)
    store.save(scene)
    report = publish_pieces(store,tmp_path/"published")
    rows = [json.loads(line) for line in (tmp_path/"published/segmentation_manifest.jsonl").read_text().splitlines()]
    assert report["approved_piece_count"]==2
    assert len({r["capture_id"] for r in rows})==2
    assert len({r["piece_path"] for r in rows})==2
    assert len({r["source_capture_id"] for r in rows})==1
    assert all(file_hash(Path(r["source_path"]))==r["source_sha256"] for r in rows)
    from universal_segmenter.release_cli import verify_output
    assert verify_output(tmp_path/"published")["status"]=="ready"
    assert not report["physical_inventory_confirmed"]


def test_export_import_duplicates_conflicts_and_partial_omission(tmp_path):
    store = SceneStore(tmp_path/"work")
    scene = completed(store)
    store.add(rgb(2),metadata(page="p2",session="s2"))
    report = export_collection(store,tmp_path/"one.zip")
    assert report["unfinished_omitted"]==1
    first = import_collection(tmp_path/"one.zip",tmp_path/"central")
    second = import_collection(tmp_path/"one.zip",tmp_path/"central")
    assert len(first["accepted"])==1 and second["already_imported"]
    edit_instance(scene,scene.active[0]["instance_id"],mask(15),accept=True)
    finish_scene(scene); store.save(scene)
    export_collection(store,tmp_path/"two.zip")
    changed = import_collection(tmp_path/"two.zip",tmp_path/"central")
    assert changed["conflicts"]==[scene.record["capture_id"]]
    index = read_json(tmp_path/"central/capture_index.json")
    assert index[scene.record["capture_id"]]["collection_id"]==first["collection_id"]


def test_zip_tampering_and_path_escape_are_rejected(tmp_path):
    store = SceneStore(tmp_path/"work")
    completed(store)
    export_collection(store,tmp_path/"one.zip")
    with zipfile.ZipFile(tmp_path/"one.zip") as src, zipfile.ZipFile(tmp_path/"bad.zip","w") as dst:
        for name in src.namelist():
            dst.writestr(name,b"broken" if name.endswith(".npz") else src.read(name))
    with pytest.raises(ValueError,match="checksum"):
        import_collection(tmp_path/"bad.zip",tmp_path/"central")
    with zipfile.ZipFile(tmp_path/"escape.zip","w") as dst:
        dst.writestr("../outside.txt",b"no")
    with pytest.raises(ValueError,match="escapes"):
        import_collection(tmp_path/"escape.zip",tmp_path/"central")
    assert not (tmp_path/"outside.txt").exists()


def test_grouped_splits_preserve_prior_assignments_and_reject_conflicts(tmp_path):
    scenes = [new_scene(rgb(i),metadata(page="p"+str(i),session="shared")) for i in range(1,5)]
    split = assign_splits(scenes,tmp_path/"ledger.json")
    assert len(set(split.values()))==1
    before = read_json(tmp_path/"ledger.json")
    assert assign_splits(scenes[::-1],tmp_path/"ledger.json")==split
    assert read_json(tmp_path/"ledger.json")==before
    scenes[0].record["metadata"]["split"]="train"
    scenes[1].record["metadata"]["split"]="holdout"
    with pytest.raises(ValueError,match="cross protected"):
        assign_splits(scenes,tmp_path/"another.json")


def test_dataset_freeze_native_letterbox_tiles_and_empty_loss(tmp_path):
    store = SceneStore(tmp_path/"work")
    completed(store,split="train")
    empty = store.add(rgb(2),metadata(page="p2",session="s2",split="train"))
    finish_scene(empty,empty=True); store.save(empty)
    report = build_dataset([store.root],tmp_path/"dataset",tmp_path/"ledger.json")
    assert report["sample_count"]==2 and not report["evaluation_ready"]
    assert verify_dataset(tmp_path/"dataset")["sample_count"]==2
    from universal_segmenter.scenes.dataset import _legacy_rows
    rows = _legacy_rows(tmp_path/"dataset")
    positive = next(r for r in rows if r["sample_kind"]=="real_scene")
    image,labels,transform = load_view(tmp_path/"dataset",positive,96)
    assert image.shape==(96,96,3) and len(np.unique(labels))==3
    assert transform["letterbox"]["top"]>0
    _,tile,_ = load_view(tmp_path/"dataset",positive,64,(10,10,30,30))
    assert tile.any()  # Partial pieces at internal tile edges remain labels.
    import torch
    from universal_segmenter.training.dataset import PaperDataset
    from universal_segmenter.losses.paper_loss import create_paper_loss
    loader = PaperDataset(tmp_path/"dataset",rows,96,training=False)
    for i in range(len(rows)):
        batch = loader[i]
        outputs={"mask_logits":torch.zeros((1,1,96,96),requires_grad=True),"distance":torch.zeros((1,1,96,96))}
        loss=create_paper_loss()(outputs,batch["mask"][None],batch["distance"][None])["loss"]
        assert torch.isfinite(loss)
        loss.backward()
    (tmp_path/"dataset/samples.csv").write_text("tampered")
    with pytest.raises(ValueError,match="changed"):
        verify_dataset(tmp_path/"dataset")


def test_invalid_accepted_masks_and_label_ids():
    scene = new_scene(rgb(),metadata())
    clipped = mask(); clipped[:15,10:30]=True
    with pytest.raises(ValueError,match="clipped"):
        add_instance(scene,clipped,accepted=True)
    add_instance(scene,mask(),accepted=True)
    add_instance(scene,mask(60),accepted=True)
    scene.record["instances"][1]["label"]=1
    with pytest.raises(ValueError,match="unique"):
        validate_scene(scene)


def test_instance_metrics_empty_and_missed_small_piece():
    target=np.zeros((20,20),np.int32); target[2:10,2:10]=1; target[15:17,15:17]=2
    predicted=target.copy(); predicted[predicted==2]=0
    result=instance_metrics(predicted,target)
    assert result["recall"]==.5 and not result["exact_count"]
    assert instance_metrics(target*0,target*0)["precision"]==1


def test_central_conflict_resolution_drives_new_dataset_without_overwriting(tmp_path):
    from universal_segmenter.scenes.exchange import resolve_annotation
    store=SceneStore(tmp_path/"work")
    scene=completed(store,split="train")
    export_collection(store,tmp_path/"first.zip")
    first=import_collection(tmp_path/"first.zip",tmp_path/"central")
    edit_instance(scene,scene.active[0]["instance_id"],mask(15),accept=True)
    finish_scene(scene); store.save(scene)
    export_collection(store,tmp_path/"second.zip")
    second=import_collection(tmp_path/"second.zip",tmp_path/"central")
    resolve_annotation(tmp_path/"central",scene.record["capture_id"],second["collection_id"])
    build_dataset([],tmp_path/"new_dataset",tmp_path/"ledger.json",central=tmp_path/"central")
    current=SceneStore(tmp_path/"new_dataset").load(scene.record["capture_id"])
    assert np.array_equal(current.labels,scene.labels)
    assert (tmp_path/"central/collections"/first["collection_id"]).exists()


def test_content_freeze_is_training_ready_only_with_independent_real_splits(tmp_path):
    from universal_segmenter.training.freeze import verify_dataset_freeze
    store=SceneStore(tmp_path/"work")
    for seed,split in enumerate(["train","validation","calibration","holdout"],1):
        completed(store,seed=seed,page=split,session=split,split=split)
    build_dataset([store.root],tmp_path/"dataset",tmp_path/"ledger.json")
    assert verify_dataset_freeze(tmp_path/"dataset")["fingerprint"]


def test_scene_count_sampling_is_deterministic_and_balanced():
    from universal_segmenter.training.dataset import build_epoch_plan
    rows=[{"sample_id":str(n),"split":"train","sample_kind":"real_scene", "capture_profile":"phone", "instance_count":str(n)}
          for n in [1,2,4,8,16,32]]
    plan=build_epoch_plan(rows,0,123,60,{"real_capture":1},{"1":1,"2-4":1,"5-8":1,"9-16":1,"17+":1})
    assert plan==build_epoch_plan(rows,0,123,60,{"real_capture":1},{"1":1,"2-4":1,"5-8":1,"9-16":1,"17+":1})
    assert sum(rows[d.row_index]["instance_count"]=="32" for d in plan)==12


def test_stored_scene_matches_public_json_schema(tmp_path):
    import jsonschema
    store=SceneStore(tmp_path/"work")
    scene=completed(store)
    schema=read_json(Path(__file__).resolve().parents[1]/"schemas/scene.schema.json")
    jsonschema.validate(scene.record,schema)


def test_instance_label_storage_does_not_truncate_at_255(tmp_path):
    store=SceneStore(tmp_path/"work")
    scene=store.add(np.full((100,100,3),80,np.uint8),metadata())
    for i in range(256):
        region=np.zeros((100,100),bool)
        x,y=2+(i%16)*6,2+(i//16)*6
        region[y:y+2,x:x+2]=True
        add_instance(scene,region,accepted=True)
    finish_scene(scene);store.save(scene)
    restored=store.load(scene.record["capture_id"])
    assert restored.labels.max()==256 and len(restored.active)==256


def test_checked_collection_is_immutable_but_can_be_cloned_for_correction(tmp_path):
    from universal_segmenter.scenes.workspace import clone_workspace
    store=SceneStore(tmp_path/"work");completed(store)
    export_collection(store,tmp_path/"one.zip")
    receipt=import_collection(tmp_path/"one.zip",tmp_path/"central")
    checked=SceneStore(tmp_path/"central/collections"/receipt["collection_id"])
    scene=checked.load(checked.ids()[0])
    with pytest.raises(ValueError,match="immutable"):
        checked.save(scene)
    clone_workspace(checked,tmp_path/"correctable")
    mutable=SceneStore(tmp_path/"correctable")
    corrected=mutable.load(mutable.ids()[0])
    edit_instance(corrected,corrected.active[0]["instance_id"],mask(15),accept=True)
    mutable.save(corrected)
    assert np.array_equal(checked.load(checked.ids()[0]).labels,scene.labels)


def test_generated_truth_does_not_impersonate_manual_approval(tmp_path):
    store=SceneStore(tmp_path/"work")
    scene=store.add(rgb(),metadata())
    add_instance(scene,mask(),accepted=True,origin="synthetic_truth")
    finish_scene(scene);store.save(scene)
    assert publish_pieces(store,tmp_path/"normal")["approved_piece_count"]==0
    report=publish_pieces(store,tmp_path/"diagnostic",include_generated_fixtures=True)
    assert report["approved_piece_count"]==1
    row=json.loads((tmp_path/"diagnostic/segmentation_manifest.jsonl").read_text())
    assert row["decision_source"]=="synthetic_truth" and row["review_status"]=="auto_accepted"
