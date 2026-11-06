from dataclasses import replace

import numpy as np
import pytest
from PIL import Image,ImageDraw
from scipy import ndimage

from universal_segmenter.scenes.generator import GeneratorConfig, render_scene, generate, collection_assets
from universal_segmenter.scenes.core import SceneStore, file_hash, write_json


def piece():
    image=Image.new("RGBA",(90,120),(240,240,225,0))
    ImageDraw.Draw(image).polygon([(4,5),(80,4),(84,60),(65,110),(8,115)],fill=(245,245,235,255))
    ImageDraw.Draw(image).text((15,30),"TEXT",fill=(20,20,20,255))
    return image


@pytest.mark.parametrize("count",[0,1,4,12,32])
def test_scene_truth_and_reproducibility(count):
    config=GeneratorConfig(master_size=384,training_size=192,minimum_gap_px=4,effects=True)
    args=([piece() for _ in range(count)],None,"phone",config,151)
    rgb,labels,recipe=render_scene(*args)
    again,other,repeat=render_scene(*args)
    assert np.array_equal(rgb,again) and np.array_equal(labels,other) and recipe==repeat
    assert len(np.unique(labels))-1==count
    for i in range(1,count+1):
        assert ndimage.label(labels==i)[1]==1
        assert not np.any(ndimage.binary_dilation(labels==i,iterations=3)&((labels!=i)&(labels>0)))


def test_integer_labels_survive_global_camera_transform():
    config=GeneratorConfig(master_size=384,training_size=192,minimum_gap_px=5,
                           global_perspective=.005,radial_distortion=.008)
    _,labels,_=render_scene([piece()]*4,None,"phone",config,17)
    assert set(np.unique(labels))=={0,1,2,3,4}


def test_pdf_generator_resume_and_changed_sources(tmp_path):
    import pymupdf
    pdf_path=tmp_path/"page.pdf"
    with pymupdf.open() as pdf:
        page=pdf.new_page(width=240,height=320)
        page.insert_text((25,40),"Paper reconstruction generator test")
        pdf.save(pdf_path)
    bg=tmp_path/"bg.png"; Image.new("RGB",(400,400),(50,70,100)).save(bg)
    inventory=tmp_path/"inventory.json"
    write_json(inventory,{"schema_version":"paper-scene-source-inventory-1.0","assets":[
        {"kind":"background","path":str(bg),"sha256":file_hash(bg),"capture_profile":"scanner","split":"train"},
        {"kind":"pdf","path":str(pdf_path),"sha256":file_hash(pdf_path),"document_id":"doc","split":"train"}]})
    config=GeneratorConfig(scene_count=2,master_size=384,training_size=192,
        count_bins=[[4,4,1]],source_weights={"procedural":1},profile_weights={"scanner":1})
    report=generate(inventory,tmp_path/"scenes",config)
    assert report["scenes"]==2 and report["counts"]=={"4":2}
    assert generate(inventory,tmp_path/"scenes",config,resume=True)==report
    Image.new("RGB",(400,400),"red").save(bg)
    with pytest.raises(ValueError,match="changed"):
        generate(inventory,tmp_path/"scenes",config,resume=True)


def test_nontrain_assets_cannot_generate(tmp_path):
    inventory=tmp_path/"inventory.json"
    write_json(inventory,{"schema_version":"paper-scene-source-inventory-1.0","assets":[]})
    config=GeneratorConfig(scene_count=1,count_bins=[[0,0,1]],profile_weights={"scanner":1})
    with pytest.raises(ValueError,match="training-eligible"):
        generate(inventory,tmp_path/"scenes",config)


def test_reviewed_collection_cutouts_and_negatives_obey_split_ledger(tmp_path):
    from universal_segmenter.scenes.dataset import assign_splits
    from universal_segmenter.scenes.workspace import add_instance,finish_scene
    store=SceneStore(tmp_path/"work")
    for n,split in enumerate(["train","holdout"]):
        rgb=np.full((100,120,3),70+n,np.uint8)
        scene=store.add(rgb,{"document_id":"doc","page_id":split,"capture_session_id":split,"capture_profile":"phone","split":split})
        mask=np.zeros((100,120),bool); mask[10:80,20:90]=True
        add_instance(scene,mask,accepted=True); finish_scene(scene); store.save(scene)
    scenes=[store.load(i) for i in store.ids()]
    assign_splits(scenes,tmp_path/"ledger.json")
    inventory=tmp_path/"inventory.json"
    write_json(inventory,{"schema_version":"paper-scene-source-inventory-1.0","assets":[]})
    result=collection_assets(store,inventory,tmp_path/"ledger.json")
    assert result["assets_added"]==1 and result["scenes_skipped"]==1
    assert collection_assets(store,inventory,tmp_path/"ledger.json")["assets_added"]==0


def test_real_and_retextured_source_families(tmp_path):
    import pymupdf
    from universal_segmenter.scenes.generator import _pieces
    pdf_path=tmp_path/"texture.pdf"
    with pymupdf.open() as pdf:
        page=pdf.new_page(width=200,height=300)
        page.insert_text((20,30),"texture")
        pdf.save(pdf_path)
    rgba=piece(); source=tmp_path/"real.png"; mask=tmp_path/"mask.png"
    rgba.convert("RGB").save(source); rgba.getchannel("A").save(mask)
    assets=[{"kind":"real","path":str(source),"mask_path":str(mask),"split":"train"},
            {"kind":"pdf","path":str(pdf_path),"split":"train"}]
    config=GeneratorConfig(master_size=384,training_size=192)
    for kind in ["real","retextured","mixed"]:
        pieces,used=_pieces(4,kind,assets,config,np.random.default_rng(15))
        assert len(pieces)==4 and used
        _,labels,_=render_scene(pieces,None,"scanner",config,22)
        assert set(np.unique(labels))=={0,1,2,3,4}
