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

from universal_segmenter.annotation.app import launch_annotation_tool


if __name__ == "__main__":
    launch_annotation_tool(PROJECT)

