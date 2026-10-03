"""Build a dependency-free onedir annotation application for this OS.

Most users should run ``scripts/export_standalone_annotator.py``.  This module
contains the lower-level, importable build function so the exporter and tests
share exactly the same packaging rules.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from universal_segmenter.annotation.ml_proposals import load_portable_model


# USER SETTINGS
APPLICATION_NAME = "PaperSegmentationAnnotator"
BUILD_ROOT = PROJECT / "build" / "portable_annotator"
DIST_ROOT = PROJECT / "dist"
RELEASE_ROOT = PROJECT / "release"

# Keep feature-critical modules explicit.  PyInstaller normally discovers
# these imports from annotation.app, but an explicit entry makes the release
# contract robust if the launcher later switches to lazy imports.
FEATURE_MODULES = (
    "universal_segmenter.annotation.app",
    "universal_segmenter.annotation.editor",
    "universal_segmenter.annotation.session",
    "universal_segmenter.annotation.split_policy",
    "universal_segmenter.data.manifest",
    "universal_segmenter.data.physical_identity",
    "universal_segmenter.data.splits",
    "universal_segmenter.inference.release",
    "universal_segmenter.inference.postprocess",
    "universal_segmenter.quality.structural",
)


def build_standalone(
        project: Path = PROJECT,
        system: str | None = None,
        python_executable: str | None = None,
        ) -> Path:
    """Build and verify the standalone application, returning its ZIP path."""
    project = Path(project).resolve()
    system = system or platform.system()
    python_executable = python_executable or sys.executable
    model = load_portable_model(project)
    if system not in {"Windows", "Darwin"}:
        raise RuntimeError(
            "The supported teammate builds are Windows and macOS. Build on the "
            "same operating system that will run the application.")
    try:
        subprocess.run(
            [python_executable, "-m", "PyInstaller", "--version"],
            cwd=project,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "PyInstaller is unavailable in this environment. Run "
            "`python -m pip install -r requirements-portable-build.txt` and "
            "then rerun the exporter.") from error

    build_root = project / "build" / "portable_annotator"
    dist_root = project / "dist"
    release_root = project / "release"
    separator = ";" if system == "Windows" else ":"
    command = [
        python_executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--onedir",
        "--name", APPLICATION_NAME,
        "--distpath", str(dist_root),
        "--workpath", str(build_root / "work"),
        "--specpath", str(build_root / "spec"),
        "--paths", str(project / "src"),
        # The standard PyInstaller hook plus the native binaries are enough for
        # CPU inference. `--collect-all onnxruntime` also follows optional
        # quantization/transformer tools and can accidentally bundle the full
        # training environment (Torch, Transformers, OpenCV, pytest, ...).
        "--collect-binaries", "onnxruntime",
        "--hidden-import", "onnxruntime",
        "--hidden-import", "onnxruntime.capi._pybind_state",
        "--hidden-import", "onnxruntime.capi.onnxruntime_inference_collection",
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "transformers",
        "--exclude-module", "onnx",
        "--exclude-module", "cv2",
        "--exclude-module", "pytest",
        "--hidden-import", "tkinter",
        "--hidden-import", "tkinter.ttk",
        "--hidden-import", "tkinter.filedialog",
        "--hidden-import", "tkinter.messagebox",
        "--hidden-import", "PIL.ImageTk",
        "--hidden-import", "pymupdf",
    ]
    for module in FEATURE_MODULES:
        command.extend(("--hidden-import", module))
    for distribution in (
            "numpy", "scipy", "Pillow", "PyMuPDF", "onnxruntime"):
        command.extend(("--copy-metadata", distribution))
    command.extend((
        "--add-data",
        f"{project / 'docs' / 'THIRD_PARTY_LICENSES.md'}{separator}.",
    ))
    for path in (model.model_path, model.metadata_path):
        command.extend(("--add-data", f"{path}{separator}models"))
    command.append(str(project / "scripts" / "run_annotator.py"))
    print("[build] Running PyInstaller for", system, flush=True)
    subprocess.run(command, cwd=project, check=True)

    release_name = (
        f"{APPLICATION_NAME}-Windows-x86_64"
        if system == "Windows"
        else f"{APPLICATION_NAME}-macOS-{platform.machine()}"
    )
    release_root.mkdir(parents=True, exist_ok=True)
    archive_base = release_root / release_name
    archive = archive_base.with_suffix(".zip")
    archive.unlink(missing_ok=True)
    built = dist_root / APPLICATION_NAME
    if system == "Darwin":
        app = dist_root / f"{APPLICATION_NAME}.app"
        if app.is_dir():
            built = app
    if not built.exists():
        raise FileNotFoundError(f"PyInstaller output is missing: {built}")
    internal = built / "_internal" if built.is_dir() else built
    accidentally_bundled = [
        name for name in ("torch", "torchvision", "transformers", "cv2", "pytest")
        if (internal / name).exists()
    ]
    if accidentally_bundled:
        raise RuntimeError(
            "Portable build accidentally contains training-only packages: "
            + ", ".join(accidentally_bundled))
    bundled_names = {path.name for path in built.rglob("*") if path.is_file()}
    missing_resources = [
        path.name
        for path in (model.model_path, model.metadata_path)
        if path.name not in bundled_names
    ]
    if missing_resources:
        raise RuntimeError(
            "Portable build is missing required model resources: "
            + ", ".join(missing_resources))
    print("[build] Compressing the standalone folder into a ZIP...", flush=True)
    shutil.make_archive(
        str(archive_base), "zip", root_dir=built.parent, base_dir=built.name)
    print(f"[build] Complete: {archive}", flush=True)
    print(
        "[build] Extract the ZIP before running it. Build Windows on Windows "
        "and macOS on macOS; PyInstaller does not cross-compile.",
        flush=True,
    )
    return archive


def main() -> None:
    build_standalone()


if __name__ == "__main__":
    main()
