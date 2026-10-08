"""One loader for scanner PDFs and ordinary image captures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    original_hw: tuple[int, int]
    canvas_hw: tuple[int, int]
    left: int
    top: int
    resized_width: int
    resized_height: int


def load_rgb(path: Path, pdf_dpi: int = 300) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        try:
            import pymupdf
        except ImportError as error:
            raise RuntimeError("PDF input requires the optional PyMuPDF dependency") from error
        document = pymupdf.open(path)
        try:
            if document.page_count != 1:
                raise ValueError(f"expected one PDF page, found {document.page_count}: {path}")
            scale = pdf_dpi / 72.0
            pixmap = document[0].get_pixmap(
                matrix=pymupdf.Matrix(scale, scale), alpha=False)
            array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n)
            return np.ascontiguousarray(array[:, :, :3])
        finally:
            document.close()
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return np.array(image, dtype=np.uint8, copy=True)


def letterbox(rgb: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxTransform]:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("letterbox expects HxWx3 uint8 RGB")
    height, width = rgb.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = Image.fromarray(rgb).resize(
        (resized_width, resized_height), Image.Resampling.LANCZOS)
    band = max(1, round(min(height, width) * 0.025))
    border = np.concatenate((
        rgb[:band].reshape(-1, 3), rgb[-band:].reshape(-1, 3),
        rgb[:, :band].reshape(-1, 3), rgb[:, -band:].reshape(-1, 3)))
    fill = tuple(int(value) for value in np.median(border, axis=0))
    canvas = Image.new("RGB", (size, size), fill)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas.paste(resized, (left, top))
    transform = LetterboxTransform(
        (height, width), (size, size), left, top, resized_width, resized_height)
    return np.asarray(canvas, dtype=np.uint8), transform


def restore_probability(probability: np.ndarray,
                        transform: LetterboxTransform) -> np.ndarray:
    crop = probability[
        transform.top:transform.top + transform.resized_height,
        transform.left:transform.left + transform.resized_width]
    height, width = transform.original_hw
    restored = np.asarray(
        Image.fromarray(crop.astype(np.float32), mode="F").resize(
            (width, height), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    if not np.isfinite(restored).all():
        raise ValueError("restored probability map contains NaN or infinity")
    # Bilinear float32 restoration can also move a saturated endpoint by a
    # few ULPs.  Keep the final map inside its documented probability domain.
    return np.clip(restored, 0.0, 1.0).astype(np.float32, copy=False)
