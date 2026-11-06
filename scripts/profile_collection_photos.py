"""Measure real photo collection latency without altering input files or existing workspaces."""
import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"src"))

from universal_segmenter.inference.release import Segmenter
from universal_segmenter.scenes.core import SceneStore,write_json
from universal_segmenter.scenes.workspace import propose


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photos",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new diagnostics folder")
    store=SceneStore(args.output/"workspace")
    started=perf_counter();segmenter=Segmenter(project=PROJECT,cpu_threads=4)
    model_load=perf_counter()-started
    rows=[]
    for path in sorted(args.photos.glob("*.jpg")):
        started=perf_counter()
        with Image.open(path) as image:
            rgb=np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        decoded=perf_counter()
        scene=store.add(rgb,{"capture_profile":"phone","source_name":path.name})
        imported=perf_counter()
        probability,_,timings=segmenter.probability(rgb)
        inferred=perf_counter()
        propose(scene,probability,segmenter.release.threshold,segmenter.release.cache_namespace)
        extracted=perf_counter()
        store.save(scene)
        saved=perf_counter()
        # Model-only evaluation is not an accuracy claim: annotations remain proposed.
        store.load(scene.record["capture_id"])
        loaded=perf_counter()
        row={"source_name":path.name,"native_hw":list(rgb.shape[:2]),"pieces_proposed":len(scene.active),
             "timings_seconds":{"decode":decoded-started,"first_import":imported-decoded,
              "probability":inferred-imported,"inference":timings["inference"]/1000,
              "separation":extracted-inferred,"save":saved-extracted,"reload":loaded-saved,
              "suggestion_total":saved-imported}}
        rows.append(row);print(row,flush=True)
    write_json(args.output/"profile.json",{"cpu_threads":4,"model_load_seconds":model_load,
        "model_namespace":segmenter.release.cache_namespace,"accuracy_verified":False,"photos":rows})


if __name__=="__main__":
    main()
