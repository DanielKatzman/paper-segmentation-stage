"""Scan the three input folders and build the approved synthetic/real dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from universal_segmenter.generation import GeneratorConfig, generate_dataset, sync_generator_inputs


INPUT_ROOT = Path("data/generator_inputs")
OUTPUT_ROOT = Path("data/complete_dataset_v3")
RANDOM_SEED = 73191


def main() -> None:
    output = PROJECT / OUTPUT_ROOT
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"{output} is not empty. Keep the completed dataset, or remove only "
            "an unfinished output folder and rerun.")
    print("[dataset] Scanning backgrounds, PDFs, images, masks, and copied "
          "annotator manifests...", flush=True)
    inventory = sync_generator_inputs(PROJECT / INPUT_ROOT)
    print(json.dumps(inventory, indent=2))
    config = GeneratorConfig(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        random_seed=RANDOM_SEED,
        image_size=768,
        pdf_dpi=120,
        max_pages_per_pdf=5,
        puzzles_per_pdf_page=2,
        min_pieces_per_puzzle=4,
        max_pieces_per_puzzle=8,
        geometry_raster_attempts=8,
        puzzle_render_attempts=8,
        capture_pose_attempts=24,
        variants_per_synthetic_piece=1,
        real_composites_per_piece=12,
        real_retextures_per_piece=4,
        background_negative_ratio=0.05,
        preview_count=30,
        require_real_backgrounds=True,
        show_progress=True,
        progress_update_seconds=1.0,
    )
    report = generate_dataset(PROJECT, config)
    print(json.dumps(report, indent=2))
    print("\nDATASET VALID:", report["valid"])
    print("Next: inspect data/complete_dataset_v3/previews, then run "
          "scripts/freeze_dataset.py")


if __name__ == "__main__":
    main()

