"""Native-scale per-piece publication for photos containing separated fragments."""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage

from universal_segmenter.data.manifest import read_jsonl, sha256_file, sync_capture_manifest, write_jsonl
from universal_segmenter.inference.preprocess import load_rgb
from universal_segmenter.inference.release import Segmenter
from universal_segmenter.quality.structural import StructuralDiagnostics, structural_reasons
from universal_segmenter.scenes.core import Scene, atomic_bytes, validate_scene
from universal_segmenter.scenes.workspace import PROPOSAL_POLICY
from .runner import _atomic_json, _bbox, _existing_is_current, _materialize


def _prediction(segmenter, output, source_hash, rgb, minimum_pixels):
    namespace = f"{segmenter.release.cache_namespace}-{PROPOSAL_POLICY}-min{minimum_pixels}"
    path = output / "cache" / namespace / (source_hash+".npz")
    metadata = path.with_suffix(".json")
    if path.is_file() and metadata.is_file():
        try:
            record = json.loads(metadata.read_text(encoding="utf-8"))
            if record["cache_sha256"] != sha256_file(path):
                raise ValueError("Prediction cache checksum mismatch")
            with np.load(path,allow_pickle=False) as arrays:
                scene = Scene(record["scene"],arrays["labels"].copy())
            validate_scene(scene)
            if scene.labels.shape != rgb.shape[:2]:
                raise ValueError("Prediction cache shape differs")
            return scene, True
        except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile):
            pass
    scene = segmenter.segment_scene(rgb,minimum_pixels=minimum_pixels)
    stream = io.BytesIO()
    np.savez_compressed(stream,labels=scene.labels)
    atomic_bytes(path,stream.getvalue())
    _atomic_json(metadata,{"scene":scene.record,"cache_sha256":sha256_file(path)})
    return scene, False


def _diagnostics(mask, box):
    x,y,w,h = box
    cropped = mask[y:y+h,x:x+w]
    holes = ndimage.binary_fill_holes(cropped) & ~cropped
    return StructuralDiagnostics(float(mask.mean()),
        int(ndimage.label(cropped,np.ones((3,3),bool))[1]),0.,float(holes.sum()/mask.size),
        bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()))


def run_multi_pipeline(input_folder, output_folder, review="flagged", cpu_threads=None,
                       pdf_dpi=300, reprocess=False, project=None, headless=False,
                       release_directory=None, minimum_pixels=4):
    if review not in {"none","flagged","all"} or minimum_pixels < 1:
        raise ValueError("Invalid review mode or minimum component area")
    source,output = Path(input_folder).resolve(),Path(output_folder).resolve()
    if not source.is_dir() or source==output or source in output.parents or output in source.parents:
        raise ValueError("Input must exist and output must not overlap it")
    output.mkdir(parents=True,exist_ok=True)
    capture_manifest = output/".state/captures.jsonl"
    scan = sync_capture_manifest(source,capture_manifest,output)
    captures = read_jsonl(capture_manifest)
    segmenter = Segmenter(project=project,cpu_threads=cpu_threads,release_directory=release_directory)
    version = f"{segmenter.release.model_version}:{segmenter.release.model_sha256[:16]}"
    manifest = output/"segmentation_manifest.jsonl"
    existing = {r["capture_id"]:r for r in read_jsonl(manifest)}
    rows,sources = [],[]
    generated = reused = manual = 0
    stopped = False

    def publish_report(complete):
        write_jsonl(manifest,sorted(rows,key=lambda r:r["capture_id"]))
        report = {"schema_version":"paper-segmentation-run-1.0",
            "created_utc":datetime.now(timezone.utc).isoformat(),"complete":complete,
            "stopped_safely":stopped,"input_folder":str(source),"output_folder":str(output),
            "review_mode":review,"model_version":version,"folder_scan":scan,
            "capture_mode":"multi","proposal_policy":PROPOSAL_POLICY,"minimum_pixels":minimum_pixels,
            "total_discovered":len(captures),"source_count":len(sources),"sources":sources,
            "piece_count":sum(not r.get("empty_proposal",False) for r in rows),
            "materialized_this_run":generated,"reused_complete_outputs":reused,
            "manually_reviewed_this_run":manual,"needs_review":sum(not r["accepted"] for r in rows),
            "manifest":str(manifest)}
        _atomic_json(output/"segmentation_run.json",report)
        return report

    for number,capture in enumerate(captures,1):
        source_hash = str(capture["source_sha256"])
        source_id = str(capture["capture_id"])
        path = Path(capture["image_path"])
        if not path.is_absolute():
            path = output/path
        rgb = load_rgb(path,pdf_dpi=pdf_dpi)
        scene,cache_reused = _prediction(segmenter,output,source_hash,rgb,minimum_pixels)
        # Keep an explicit pending row for a zero-proposal image so an empty
        # prediction cannot silently certify that a physical fragment is absent.
        candidates = scene.active or [None]
        observations = []
        for item in candidates:
            ident = source_id if len(candidates)==1 else source_id+"_"+item["instance_id"]
            previous = existing.get(ident)
            if (not reprocess and previous and previous.get("postprocess_policy")==PROPOSAL_POLICY
                    and _existing_is_current(previous,output,source_hash,version)):
                rows.append(previous);observations.append(ident);reused += 1
                continue
            mask = scene.labels==item["label"] if item else np.zeros(rgb.shape[:2],bool)
            auto_mask = mask.copy()
            box = _bbox(mask)
            diagnostics = _diagnostics(mask,box)
            reasons = [str(r) for r in structural_reasons(diagnostics)]
            should_review = review=="all" or (review=="flagged" and bool(reasons))
            decision = "automatic"
            if should_review and not headless:
                from universal_segmenter.annotation.editor import MaskEditor
                editor = MaskEditor(rgb,mask,auto_mask,
                    f"Photo {number}/{len(captures)} — piece {len(observations)+1}/{len(candidates)}",
                    auto_regularize=False,keep_largest_on_save=False)
                action,corrected = editor.run()
                if action=="quit":
                    stopped=True
                    break
                box = _bbox(corrected)
                diagnostics = _diagnostics(corrected,box)
                if (not corrected.any() or diagnostics.component_count!=1
                        or diagnostics.touches_border or corrected.all()):
                    raise ValueError("Correction must contain one unclipped connected piece")
                mask,decision,reasons = corrected,"manual_review",[]
                manual += 1
            if headless and should_review:
                reasons = sorted(set(reasons+["MANUAL_REVIEW_REQUIRED"]))
            accepted = not reasons
            artifact_id = source_hash if len(candidates)==1 else ident
            paths = _materialize(output,artifact_id,rgb,mask)
            row = {"schema_version":"paper-segmentation-manifest-1.0","capture_id":ident,
                "source_capture_id":source_id,"candidate_id":item["instance_id"] if item else "empty",
                "artifact_id":artifact_id,"source_path":str(path.resolve()),"source_sha256":source_hash,
                "capture_type_hint":capture.get("capture_type","unknown"),
                "original_size_hw":list(rgb.shape[:2]),"bbox_xywh":list(_bbox(mask)),
                "mask_path":paths["mask"],"piece_path":paths["piece"],
                "piece_mask_path":paths["piece_mask"],"overlay_path":paths["overlay"],
                "model_release":segmenter.release.model_version,"model_version":version,
                "model_sha256":segmenter.release.model_sha256,
                "dataset_fingerprint":segmenter.release.dataset_fingerprint,
                "threshold":segmenter.release.threshold,"postprocess_policy":PROPOSAL_POLICY,
                "decision_source":decision,"accepted":accepted,"reason_codes":reasons,
                "review_status":"manually_reviewed" if decision=="manual_review" else
                    "auto_accepted" if accepted else "needs_review",
                "manual_changed_fraction":float(np.mean(mask!=auto_mask)),
                "diagnostics":asdict(diagnostics),"empty_proposal":not bool(mask.any()),
                "cache_reused":cache_reused,"created_utc":datetime.now(timezone.utc).isoformat()}
            rows.append(row);observations.append(ident);generated += 1
        sources.append({"source_capture_id":source_id,"source_sha256":source_hash,
                        "observation_ids":observations,"proposal_count":len(scene.active),
                        "empty_reviewed":False,"processed":not stopped})
        if stopped:
            break
        publish_report(False)
        print(f"[segment] photo {number}/{len(captures)}: {len(scene.active)} separate proposals",flush=True)
    return publish_report(not stopped and len(sources)==len(captures))
