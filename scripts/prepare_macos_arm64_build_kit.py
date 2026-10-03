"""Create a self-contained Apple-Silicon build kit on the Windows PC.

This does not cross-compile the application. It packages the verified ONNX
models and the minimal source tree with a double-clickable macOS build script.
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from universal_segmenter.annotation.ml_proposals import load_portable_model


KIT_NAME = "PaperSegmentationAnnotator-macOS-arm64-build-kit"
OUTPUT = PROJECT / "release" / f"{KIT_NAME}.zip"
MAC_BUILD_COMMAND = r'''#!/bin/bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

pause_on_error() {
  status=$?
  echo
  echo "The macOS build failed. Copy the complete Terminal output back to Daniel."
  echo "Press Return to close this window."
  read -r
  exit "$status"
}
trap pause_on_error ERR

echo "Paper Segmentation Annotator - Apple Silicon builder"
echo "=================================================="

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: This file must be run on macOS."
  false
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: This kit requires a native Apple Silicon Terminal (arm64)."
  echo "Do not run Terminal through Rosetta."
  false
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: Python 3 is missing. Install the current Python 3.12 macOS"
  echo "installer from https://www.python.org/downloads/macos/ and run this again."
  false
fi

python3 - <<'PY'
import platform
import sys
if platform.machine() != "arm64":
    raise SystemExit("Python is not running natively as arm64")
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required")
print(f"Using Python {platform.python_version()} ({platform.machine()})")
PY

echo
echo "[1/4] Creating an isolated build environment..."
python3 -m venv .mac-build-venv
source .mac-build-venv/bin/activate

echo "[2/4] Installing CPU-only packaging dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements-portable-build.txt

echo "[3/4] Building the native arm64 application..."
python scripts/export_standalone_annotator.py --local-only

echo "[4/4] Verifying the final package..."
RESULT="release/PaperSegmentationAnnotator-macOS-arm64.zip"
if [[ ! -f "$RESULT" ]]; then
  echo "ERROR: Expected package is missing: $RESULT"
  false
fi

echo
echo "BUILD COMPLETE"
echo "Final file: $ROOT/$RESULT"
echo "Send that ZIP back to Daniel."
open "$ROOT/release"
echo
echo "Press Return to close this window."
read -r
'''

MAC_README = """Paper Segmentation Annotator - Apple Silicon build kit

This ZIP was prepared on Windows, but the .app must be built on macOS.

1. Extract this entire ZIP on the Apple-Silicon Mac.
2. Double-click BUILD_MAC_APP.command.
3. If macOS asks, allow Terminal to run it.
4. Wait for BUILD COMPLETE.
5. Send release/PaperSegmentationAnnotator-macOS-arm64.zip back to Daniel.

If double-clicking reports a permission problem, open Terminal in this folder
and run:

    chmod +x BUILD_MAC_APP.command
    ./BUILD_MAC_APP.command

The Mac needs internet access for the first dependency installation and a
native arm64 Python 3.10 or newer. Python 3.12 is recommended.
"""


def _source_files(path: Path):
    if path.is_file():
        yield path
        return
    for item in sorted(path.rglob("*")):
        if (item.is_file()
                and "__pycache__" not in item.parts
                and item.suffix not in {".pyc", ".pyo"}):
            yield item


def _write_text(archive: zipfile.ZipFile, relative: str, text: str,
                executable: bool = False) -> None:
    info = zipfile.ZipInfo(f"{KIT_NAME}/{relative}")
    info.create_system = 3
    info.external_attr = ((0o755 if executable else 0o644) & 0xFFFF) << 16
    archive.writestr(info, text.encode("utf-8"))


def create_build_kit(project: Path = PROJECT, output: Path = OUTPUT) -> Path:
    project = Path(project).resolve()
    output = Path(output).resolve()
    source_paths = (
        project / "src" / "universal_segmenter",
        project / "scripts" / "run_annotator.py",
        project / "scripts" / "build_annotation_tool.py",
        project / "scripts" / "export_standalone_annotator.py",
        project / "pyproject.toml",
        project / "requirements-portable-build.txt",
        project / "docs" / "THIRD_PARTY_LICENSES.md",
    )
    model = load_portable_model(project)
    missing_sources = [path for path in source_paths if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(
            "Required build-kit source is missing: "
            + ", ".join(str(path) for path in missing_sources))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    print("[mac-kit] Packaging source and verified ONNX models...", flush=True)
    try:
        with zipfile.ZipFile(
                temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for source_root in source_paths:
                for source in _source_files(source_root):
                    relative = source.relative_to(project).as_posix()
                    archive.write(source, f"{KIT_NAME}/{relative}")
            for source in (model.model_path, model.metadata_path):
                print(f"[mac-kit] Adding {source.name}...", flush=True)
                archive.write(source, f"{KIT_NAME}/models/{source.name}")
            _write_text(
                archive, "BUILD_MAC_APP.command", MAC_BUILD_COMMAND,
                executable=True)
            _write_text(archive, "README_FIRST.txt", MAC_README)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[mac-kit] Complete: {output}", flush=True)
    print(
        "[mac-kit] Send this ZIP to the Apple-Silicon teammate. They extract "
        "it and double-click BUILD_MAC_APP.command.",
        flush=True,
    )
    return output


if __name__ == "__main__":
    create_build_kit()
