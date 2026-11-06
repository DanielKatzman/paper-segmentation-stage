"""Seeded multi-fragment scenes with explicit train-asset eligibility."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from universal_segmenter.generation.capture import (
    _add_shadow, _load_background, _postprocess, load_real_piece,
    piece_from_page, retexture_real_shape,
)
from universal_segmenter.generation.geometry import generate_partition, rasterize_partition
from universal_segmenter.generation.sources import discover_backgrounds, discover_pdfs, discover_real_pieces

from .core import SceneStore, digest, file_hash, read_json, write_json
from .workspace import add_instance, finish_scene
from .exchange import _contact_sheet


@dataclass
class GeneratorConfig:
    scene_count: int = 100
    master_size: int = 1536
    training_size: int = 768
    seed: int = 73191
    count_bins: list[list[float]] = field(default_factory=lambda:
        [[0, 0, .10], [1, 1, .15], [2, 4, .30], [5, 8, .25], [9, 16, .15], [17, 32, .05]])
    source_weights: dict[str, float] = field(default_factory=lambda:
        {"procedural": .45, "real": .30, "retextured": .15, "mixed": .10})
    profile_weights: dict[str, float] = field(default_factory=lambda: {"phone": .5, "scanner": .5})
    minimum_gap_px: int = 8
    minimum_piece_pixels: int = 40
    placement_attempts: int = 100
    scene_attempts: int = 12
    pdf_dpi: int = 160
    effects: bool = True
    global_perspective: float = 0.0
    radial_distortion: float = 0.0
    layout: str = "mixed"

    def validate(self) -> None:
        if (self.scene_count < 1 or self.master_size < 128 or self.training_size < 32
                or self.training_size > self.master_size or self.minimum_gap_px < 1
                or self.minimum_piece_pixels < 1 or min(self.scene_attempts, self.placement_attempts) < 1
                or not 36 <= self.pdf_dpi <= 600):
            raise ValueError("Invalid generator dimensions, counts, gap or retry limits")
        if self.master_size**2 > 60_000_000:
            raise ValueError("Master scene exceeds the 60-million-pixel memory budget")
        if not 0 <= self.global_perspective <= .05 or abs(self.radial_distortion) > .06:
            raise ValueError("Camera geometry exceeds supported augmentation bounds")
        if self.layout not in {"cells", "scatter", "mixed"}:
            raise ValueError("Layout must be cells, scatter or mixed")
        for low, high, weight in self.count_bins:
            if int(low) != low or int(high) != high or not 0 <= low <= high or weight < 0:
                raise ValueError("Count bins must be [minimum, maximum, nonnegative weight]")
        if not self.count_bins or sum(x[2] for x in self.count_bins) <= 0:
            raise ValueError("At least one positive count-bin weight is required")
        for weights, allowed in ((self.source_weights, {"procedural", "real", "retextured", "mixed"}),
                                 (self.profile_weights, {"phone", "scanner"})):
            if set(weights) - allowed or any(v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
                raise ValueError("Invalid generator mixture")


def create_inventory(input_root: Path, output: Path, *, declare_training_assets: bool = False) -> dict[str, Any]:
    """Discovery is read-only; assigning PDFs/backgrounds to train is explicit."""
    input_root = Path(input_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Source inventory exists; edit/review it instead of overwriting")
    assets = []
    for bg in discover_backgrounds(input_root / "backgrounds"):
        assets.append({"kind": "background", "path": str(bg.path.resolve()), "sha256": bg.sha256,
            "capture_profile": bg.capture_type, "session_id": bg.session_id,
            "split": "train" if declare_training_assets else "unassigned"})
    for pdf in discover_pdfs(input_root / "pdfs"):
        assets.append({"kind": "pdf", "path": str(pdf.path.resolve()), "sha256": pdf.sha256,
            "document_id": pdf.document_id, "split": "train" if declare_training_assets else "unassigned"})
    for real in discover_real_pieces(input_root / "real_pieces"):
        if real.canonical_for_compositing:
            assets.append({"kind": "real", "path": str(real.image_path.resolve()),
                "mask_path": str(real.mask_path.resolve()), "sha256": real.image_sha256,
                "mask_sha256": real.mask_sha256, "document_id": real.document_id,
                "page_id": real.page_id, "physical_piece_id": real.piece_id, "split": real.split})
    result = {"schema_version": "paper-scene-source-inventory-1.0", "assets": assets}
    write_json(output, result)
    return {"assets": len(assets), "inventory": str(output)}


def _assets(inventory: Path) -> list[dict[str, Any]]:
    value = read_json(inventory)
    if value.get("schema_version") != "paper-scene-source-inventory-1.0":
        raise ValueError("Unsupported source inventory")
    result = []
    for asset in value["assets"]:
        if asset.get("split") != "train":
            continue
        if asset["kind"] not in {"pdf", "background", "real"} or file_hash(Path(asset["path"])) != asset["sha256"]:
            raise ValueError("Unknown or changed generator source")
        if asset["kind"] == "real" and file_hash(Path(asset["mask_path"])) != asset["mask_sha256"]:
            raise ValueError("Real cutout mask changed")
        result.append(asset)
    return result


def collection_assets(store: SceneStore, inventory: Path, ledger_path: Path) -> dict[str,int]:
    """Make only centrally assigned training annotations available as cutouts/backgrounds."""
    from .core import atomic_bytes, png_bytes
    from .dataset import group_tokens
    store.verify()
    value = read_json(inventory)
    if value.get("schema_version") != "paper-scene-source-inventory-1.0":
        raise ValueError("Unsupported source inventory")
    ledger = read_json(ledger_path)
    tokens = ledger["tokens"]
    assets = list(value["assets"])
    seen = {digest(a) for a in assets}
    added = skipped = 0
    for ident in store.ids():
        scene = store.load(ident)
        keys = group_tokens(scene)
        if not scene.eligible or any(tokens.get(t) != "train" for t in keys):
            skipped += 1
            continue
        rgb = store.rgb(scene)
        m = scene.record["metadata"]
        folder = Path(inventory).resolve().parent / "reviewed_sources" / ident
        image_path = folder / "source.png"
        atomic_bytes(image_path,png_bytes(rgb))
        proposed = []
        if scene.record.get("clean_background"):
            proposed.append({"kind":"background","path":str(image_path),"sha256":file_hash(image_path),
                             "capture_profile":m.get("capture_profile","phone"),
                             "session_id":m.get("capture_session_id",ident),"split":"train"})
        for item in scene.active:
            mask_path = folder / (item["instance_id"]+".png")
            atomic_bytes(mask_path,png_bytes((scene.labels==item["label"]).astype(np.uint8)*255))
            proposed.append({"kind":"real","path":str(image_path),"mask_path":str(mask_path),
                "sha256":file_hash(image_path),"mask_sha256":file_hash(mask_path),"split":"train",
                "document_id":m.get("document_id",""),"page_id":m.get("page_id",""),
                "physical_piece_id":item.get("physical_piece_id") or item["instance_id"],
                "capture_id":ident,"annotation_revision":scene.record["revision"]})
        for asset in proposed:
            if digest(asset) not in seen:
                assets.append(asset); seen.add(digest(asset)); added += 1
    write_json(inventory,{**value,"assets":assets})
    return {"assets_added":added,"scenes_skipped":skipped}


def _choice(rng, weights):
    keys = sorted(k for k, weight in weights.items() if weight > 0)
    p = np.asarray([weights[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _page(asset, dpi, rng):
    import pymupdf
    with pymupdf.open(asset["path"]) as pdf:
        number = int(rng.integers(0, len(pdf)))
        if pdf[number].rect.width * pdf[number].rect.height * (dpi / 72)**2 > 60_000_000:
            raise ValueError("PDF texture render exceeds the native-pixel budget")
        pix = pdf[number].get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72),
                                    colorspace=pymupdf.csRGB, alpha=False)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy(), number


def _pieces(count, kind, assets, config, rng):
    pdfs = [a for a in assets if a["kind"] == "pdf"]
    reals = [a for a in assets if a["kind"] == "real"]
    used, pieces = [], []
    if count == 0:
        return pieces, used
    if kind in {"procedural", "mixed"}:
        if not pdfs:
            raise ValueError("Procedural/mixed scenes need training-eligible PDFs")
        # Dense segmentation scenes need not cut one page into hundreds of slivers.
        # Draw fresh small partitions, preserving every source page's lineage.
        while len(pieces) < count:
            asset = pdfs[int(rng.integers(len(pdfs)))]
            page, number = _page(asset, config.pdf_dpi, rng)
            amount = min(8,count-len(pieces))
            partition = generate_partition(page.shape[1], page.shape[0], max(2,amount), rng)
            labels = rasterize_partition(partition, page.shape[1], page.shape[0])
            pieces.extend(piece_from_page(page, labels == i + 1) for i in range(amount))
            used.append({**asset, "pdf_page": number})
    if kind in {"real", "retextured", "mixed"}:
        if not reals:
            raise ValueError("Real/retextured/mixed scenes need reviewed training cutouts")
        for i in range(count):
            if kind == "mixed" and i % 2 == 0:
                continue
            asset = reals[int(rng.integers(len(reals)))]
            piece = load_real_piece(Path(asset["path"]), Path(asset["mask_path"]))
            used.append(asset)
            if kind == "retextured":
                if not pdfs:
                    raise ValueError("Retextured scenes need training-eligible texture PDFs")
                pdf = pdfs[int(rng.integers(len(pdfs)))]
                page, number = _page(pdf, config.pdf_dpi, rng)
                piece = retexture_real_shape(piece, page, rng)
                used.append({**pdf, "pdf_page": number})
            if kind == "mixed":
                pieces[i] = piece
            else:
                pieces.append(piece)
    return pieces, used


def _warp_camera(rgb, labels, perspective, radial, rng):
    size = labels.shape[0]
    metadata = {}
    if perspective:
        from universal_segmenter.generation.capture import _perspective_coefficients
        extent = float(perspective) * size
        source = [(0., 0.), (size-1., 0.), (size-1., size-1.), (0., size-1.)]
        destination = [(x + float(rng.uniform(-extent, extent)), y + float(rng.uniform(-extent, extent)))
                       for x, y in source]
        coeff = _perspective_coefficients(destination, source)
        rgb = np.asarray(Image.fromarray(rgb).transform((size, size), Image.Transform.PERSPECTIVE,
                         coeff, Image.Resampling.BICUBIC), dtype=np.uint8)
        labels = np.asarray(Image.fromarray(labels).transform((size, size), Image.Transform.PERSPECTIVE,
                            coeff, Image.Resampling.NEAREST), dtype=np.int32)
        metadata["inverse_perspective_coefficients"] = list(coeff)
    if radial:
        yy, xx = np.indices(labels.shape, dtype=float)
        center = (size - 1) / 2
        dx, dy = (xx - center) / center, (yy - center) / center
        factor = 1 + radial * (dx * dx + dy * dy)
        coords = [center + dy * factor * center, center + dx * factor * center]
        rgb = np.stack([ndimage.map_coordinates(rgb[:, :, c], coords, order=1, mode="nearest")
                        for c in range(3)], axis=-1).astype(np.uint8)
        labels = ndimage.map_coordinates(labels, coords, order=0, mode="constant", cval=0).astype(np.int32)
        metadata["radial_coefficient"] = radial
    return rgb, labels, metadata


def render_scene(pieces: list[Image.Image], background: Path | None,
                 profile: str, config: GeneratorConfig, seed: int):
    """Pack heterogeneous shapes into jittered cells, with exact per-instance truth."""
    from .core import bbox, instance_errors
    rng = np.random.default_rng(seed)
    size, count = config.master_size, len(pieces)
    rgb = _load_background(background, size, rng)
    canvas = Image.fromarray(rgb).convert("RGBA")
    labels = np.zeros((size, size), np.int32)
    transforms = []
    layout = ("scatter" if rng.random() < .5 else "cells") if config.layout == "mixed" else config.layout
    if count:
        cols = math.ceil(math.sqrt(count))
        rows = math.ceil(count / cols)
        cells = rng.permutation(rows * cols)[:count]
        margin = config.minimum_gap_px + max(3, round(size * max(config.global_perspective, abs(config.radial_distortion))))
        for i, (piece, cell) in enumerate(zip(pieces, cells), start=1):
            left, top = int(cell % cols) * size // cols, int(cell // cols) * size // rows
            right, bottom = (int(cell % cols)+1)*size//cols, (int(cell//cols)+1)*size//rows
            max_w, max_h = right-left-2*margin, bottom-top-2*margin
            if min(max_w, max_h) < 5:
                raise ValueError("Requested count/gap cannot fit the master resolution")
            if layout == "scatter":
                left,top,right,bottom = 0,0,size,size
            forbidden = ndimage.binary_dilation(labels > 0, iterations=config.minimum_gap_px)
            for attempt in range(config.placement_attempts):
                angle = float(rng.uniform(0, 360))
                original = piece.convert("RGBA")
                rotated = original.rotate(angle, Image.Resampling.BICUBIC, expand=True)
                mask_image = original.getchannel("A").rotate(angle, Image.Resampling.NEAREST, expand=True)
                scale = min(max_w / rotated.width, max_h / rotated.height) * float(rng.uniform(.40, .98))
                dims = (max(3, round(rotated.width*scale)), max(3, round(rotated.height*scale)))
                rotated = rotated.resize(dims, Image.Resampling.LANCZOS)
                mask_image = mask_image.resize(dims, Image.Resampling.NEAREST)
                mask = np.asarray(mask_image) >= 128
                if mask.sum() < config.minimum_piece_pixels or ndimage.label(mask)[1] != 1:
                    continue
                x = int(rng.integers(left+margin, right-margin-dims[0]+1))
                y = int(rng.integers(top+margin, bottom-margin-dims[1]+1))
                if np.any(forbidden[y:y+dims[1],x:x+dims[0]] & mask):
                    continue
                full = np.zeros((size, size), bool)
                full[y:y+dims[1], x:x+dims[0]] = mask
                break
            else:
                raise ValueError("Could not place a connected piece within the retry budget")
            labels[full] = i
            shadow_mask = Image.fromarray(full.astype(np.uint8)*255)
            if config.effects:
                canvas, shadow = _add_shadow(canvas, shadow_mask, profile, rng)
            else:
                shadow = {}
            rotated.putalpha(Image.fromarray(mask.astype(np.uint8)*255).filter(ImageFilter.GaussianBlur(.3)))
            canvas.alpha_composite(rotated, (x, y))
            transforms.append({"instance_label": i, "rotation_degrees": angle,
                               "scale": scale, "offset_xy": [x,y], **shadow})
    rgb = np.asarray(canvas.convert("RGB"), dtype=np.uint8)
    rgb, labels, camera = _warp_camera(rgb, labels, config.global_perspective,
                                      config.radial_distortion, rng)
    if config.effects:
        # Existing postprocess's radial mask path is boolean. Apply radial geometry
        # above to integer labels, then retain all its photometric effects below.
        rgb, _, effects = _postprocess(rgb, labels > 0, profile, rng, radial_enabled=False)
    else:
        effects = {}
    for i in range(1, count+1):
        if instance_errors(labels == i):
            raise ValueError("Camera transform clipped/disconnected a piece; resample the scene")
    derived = np.asarray(Image.fromarray(labels).resize((config.training_size, config.training_size),
                         Image.Resampling.NEAREST), dtype=np.int32)
    warnings = []
    for i in range(1,count+1):
        if not (derived == i).any() or ndimage.label(derived == i)[1] != 1:
            warnings.append(f"instance_{i}_lost_or_disconnected_at_training_resolution")
    if ndimage.label(derived > 0, np.ones((3,3),bool))[1] < count:
        warnings.append("pieces_merge_at_training_resolution")
    return rgb, labels, {"placements": transforms, "layout": layout, "camera": camera, "effects": effects,
                         "derived_warnings": warnings, "source_scale_qualified": False}


def generate(inventory: Path, output: Path, config: GeneratorConfig, *, resume: bool = False) -> dict[str, Any]:
    config.validate()
    assets = _assets(inventory)
    output = Path(output).resolve()
    import importlib.metadata
    from universal_segmenter.generation import capture,geometry
    request = {"config": asdict(config), "assets": assets, "inventory_sha256": file_hash(inventory),
               "schema_version": "paper-scene-generation-1.0", "renderer_sha256":
               digest([file_hash(Path(p)) for p in (__file__,capture.__file__,geometry.__file__)]),
               "dependencies":{name:importlib.metadata.version(name) for name in
                               ("numpy","scipy","Pillow","Shapely","PyMuPDF")}}
    if output.exists():
        if not resume or not (output / "generation_request.json").exists() or read_json(output / "generation_request.json") != request:
            raise ValueError("Output exists; resume requires identical configuration and source inventory")
    else:
        output.mkdir(parents=True)
        write_json(output / "generation_request.json", request)
    store = SceneStore(output)
    completed = {store.load(i).record["metadata"].get("generation_index") for i in store.ids()}
    weights = np.asarray([b[2] for b in config.count_bins], float)
    for index in range(config.scene_count):
        if index in completed:
            continue
        seed = int(digest([config.seed,index])[:15], 16)
        rng = np.random.default_rng(seed)
        low, high, _ = config.count_bins[int(rng.choice(len(weights), p=weights/weights.sum()))]
        count = int(rng.integers(int(low), int(high)+1))
        kind = _choice(rng, config.source_weights) if count else "empty"
        profile = _choice(rng, config.profile_weights)
        backgrounds = [a for a in assets if a["kind"] == "background" and a["capture_profile"] == profile]
        if not backgrounds:
            raise ValueError(f"No training-eligible real {profile} backgrounds; add them or change profile_weights")
        bg = backgrounds[int(rng.integers(len(backgrounds)))]
        failures = []
        for attempt in range(config.scene_attempts):
            try:
                pieces, used = _pieces(count, kind, assets, config, rng)
                rgb, labels, rendering = render_scene(pieces, Path(bg["path"]), profile, config, seed+attempt)
                break
            except ValueError as error:
                failures.append(str(error))
        else:
            raise ValueError(f"Scene {index}, {count} pieces failed: {failures[-3:]}")
        # Scene is committed only after all geometry and annotation checks pass.
        scene = store.add(rgb, {"generation_index": index, "sample_kind": "background_only" if not count else
            "synthetic_" + kind, "capture_profile": profile, "split": "train",
            "document_id": "synthetic", "page_id": "scene_"+str(index),
            "seed": seed, "recipe": {"sources": [bg]+used, "rendering": rendering,
                                        "attempt": attempt, "rejections": failures}})
        for i in range(1, count+1):
            add_instance(scene, labels == i, accepted=True, origin="synthetic_truth",
                         instance_id="piece_"+digest([seed,i])[:24])
        finish_scene(scene, empty=not count, clean_background=not count)
        store.save(scene)
        print(f"[scenes] {index+1}/{config.scene_count}: {count} pieces, {profile}, {kind}", flush=True)
    report = {**store.verify(complete_only=True), "request_sha256": digest(request),
              "counts": {}, "profiles": {}, "derived_warning_count": 0}
    for ident in store.ids():
        scene = store.load(ident)
        count = str(len(scene.active))
        report["counts"][count] = report["counts"].get(count,0)+1
        profile = scene.record["metadata"]["capture_profile"]
        report["profiles"][profile] = report["profiles"].get(profile,0)+1
        report["derived_warning_count"] += len(scene.record["metadata"]["recipe"]["rendering"]["derived_warnings"])
    write_json(output / "generation_report.json", report)
    _contact_sheet(store, output / "previews/contact_sheet.jpg")
    return report
