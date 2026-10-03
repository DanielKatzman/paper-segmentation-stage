"""Request a safe checkpoint stop for the canonical training run."""

from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
STOP_FILE = (
    PROJECT / "experiments" / "segformer_b1_detail_768_real_v3" /
    "STOP_REQUESTED")
STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
STOP_FILE.touch()
print("Safe stop requested:", STOP_FILE)

