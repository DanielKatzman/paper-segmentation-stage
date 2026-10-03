"""Source-tree launcher for the Stage-1 batch pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from universal_segmenter.pipeline.cli import main


if __name__ == "__main__":
    main()
