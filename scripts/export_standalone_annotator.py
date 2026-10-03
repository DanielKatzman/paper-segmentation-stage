"""Build the local annotator and/or an Apple-Silicon teammate build kit."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from build_annotation_tool import build_standalone
from prepare_macos_arm64_build_kit import create_build_kit
from universal_segmenter.annotation.ml_proposals import load_portable_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--local-only", action="store_true")
    target.add_argument("--macos-kit-only", action="store_true")
    target.add_argument("--all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_portable_model(PROJECT)
    print(
        f"[standalone] Verified {model.display_name}: "
        f"{model.model_path.stat().st_size / 2**20:.1f} MiB, "
        f"SHA-256 {model.model_sha256[:12]}...",
        flush=True,
    )
    outputs = []
    if not args.macos_kit_only:
        outputs.append(build_standalone(PROJECT))
    if args.macos_kit_only or args.all:
        outputs.append(create_build_kit(PROJECT))
    print("\n[standalone] Export complete")
    for output in outputs:
        print(" ", output)
    if platform.system() == "Windows" and (args.macos_kit_only or args.all):
        print(
            "The macOS ZIP is a source build kit. Your teammate extracts it "
            "on Apple Silicon and double-clicks BUILD_MAC_APP.command.")


if __name__ == "__main__":
    main()

