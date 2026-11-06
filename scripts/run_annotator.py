"""Launch the standalone-compatible SegFormer annotation application."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT = (
    Path(getattr(sys, "_MEIPASS"))
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parents[1]
)
sys.path.insert(0, str(PROJECT / "src"))

from universal_segmenter.scenes.app import launch


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--self-check", type=Path)
    args = parser.parse_args()
    if args.self_check:
        from universal_segmenter.scenes.selfcheck import run
        run(PROJECT,args.self_check)
    elif args.legacy:
        from universal_segmenter.annotation.app import launch_annotation_tool
        launch_annotation_tool(PROJECT)
    else:
        launch(PROJECT,workspace=args.workspace)

