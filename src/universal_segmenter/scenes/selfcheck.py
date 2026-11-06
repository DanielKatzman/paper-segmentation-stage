"""Headless installed/bundled runtime control; no accuracy or training claim."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .core import SceneStore, write_json
from .workspace import add_instance, finish_scene, publish_pieces
from .exchange import export_collection, import_collection
from .dataset import build_dataset, verify_dataset


def run(project: Path | None, output: Path) -> dict:
    from universal_segmenter import __version__
    from universal_segmenter.inference.release import Segmenter
    segmenter = Segmenter(project=project,cpu_threads=2)
    with tempfile.TemporaryDirectory(prefix="paper-collection-control-") as folder:
        root=Path(folder)
        store=SceneStore(root/"workspace")
        rgb=np.full((192,256,3),(55,75,95),np.uint8)
        masks=[]
        for x,y in [(20,20),(150,20),(20,110),(150,110)]:
            mask=np.zeros(rgb.shape[:2],bool); mask[y:y+50,x:x+60]=True
            rgb[mask]=(240,235,220)
            masks.append(mask)
        metadata={"contributor":"selfcheck_fixture","capture_session_id":"control",
                  "capture_profile":"scanner","split":"train"}
        scene=store.add(rgb,metadata)
        for mask in masks:
            add_instance(scene,mask,accepted=True,origin="selfcheck_fixture")
        finish_scene(scene); store.save(scene)
        empty=store.add(np.full_like(rgb,80),metadata)
        finish_scene(empty,empty=True); store.save(empty)
        predicted=segmenter.segment_scene(rgb)
        from PIL import Image
        from universal_segmenter.pipeline.multi import run_multi_pipeline
        from universal_segmenter.release_cli import verify_output
        captures=root/"captures";captures.mkdir()
        Image.fromarray(rgb).save(captures/"four_pieces.png")
        run_multi_pipeline(captures,root/"headless",review="none",headless=True,
                           cpu_threads=2,project=project)
        headless=verify_output(root/"headless")
        # This synthetic fixture is a runtime control, not an accuracy benchmark.
        # Small/implausible predictions must retain their normal review gates.
        if (headless["pieces"]!=len(predicted.active) or headless["sources"]!=1
                or not headless["complete"] or headless["status"] not in {"ready","review_required"}):
            raise RuntimeError("Headless multi-piece publication control failed")
        export_collection(store,root/"collection.zip")
        import zipfile
        with zipfile.ZipFile(root/"collection.zip") as archive:
            if len([n for n in archive.namelist() if n.startswith("outlines/")])!=2:
                raise RuntimeError("Collection is missing native outlines")
        imported=import_collection(root/"collection.zip",root/"central")
        repeated=import_collection(root/"collection.zip",root/"central")
        built=build_dataset([store.root],root/"dataset",root/"ledger.json")
        verify_dataset(root/"dataset")
        published=publish_pieces(store,root/"approved",include_generated_fixtures=True)
        if (len(imported["accepted"])!=2 or not repeated["already_imported"]
                or built["sample_count"]!=2 or published["approved_piece_count"]!=4):
            raise RuntimeError("Installed collection control failed")
        report={"status":"passed","scope":"CPU inference and data workflow; no model accuracy claim",
            "core_software_version":__version__,
            "model_sha256":segmenter.release.model_sha256,"fixture_scenes":2,"fixture_pieces":4,
            "model_proposals":len(predicted.active),"collection_round_trip":True,
            "duplicate_import":True,"dataset_freeze":True,"approved_publication":True,
            "arbitrary_fragments_without_page_ids":True,"native_outline_export":True,
            "headless_multi_piece_publication":headless}
    write_json(output,report)
    return report
