"""Unified phone/scanner rendering with exact transformed foreground masks."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy import ndimage


class CaptureGeometryError(ValueError):
    """A sampled pose is invalid and can safely be retried with another seed."""


class CaptureRenderError(ValueError):
    """All sampled poses failed; the source shape must be replaced or rejected."""


def load_real_piece(image_path: Path, mask_path: Path) -> Image.Image:
    """Extract a real fragment and decontaminate pixels outside its hard mask."""
    with Image.open(image_path) as source:
        image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8)
    with Image.open(mask_path) as source:
        mask = np.asarray(ImageOps.exif_transpose(source).convert("L"), dtype=np.uint8) >= 128
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image and mask dimensions differ: {image_path}, {mask_path}")
    if int(mask.sum()) < 50 or bool(mask.all()):
        raise ValueError(f"Real piece mask is empty, full, or too small: {mask_path}")
    ys, xs = np.nonzero(mask)
    padding = max(4, int(round(max(np.ptp(xs), np.ptp(ys)) * 0.025)))
    left, right = max(0, int(xs.min()) - padding), min(mask.shape[1], int(xs.max()) + padding + 1)
    top, bottom = max(0, int(ys.min()) - padding), min(mask.shape[0], int(ys.max()) + padding + 1)
    cropped_image = image[top:bottom, left:right].copy()
    cropped_mask = mask[top:bottom, left:right]

    # Resampling consults pixels just outside the mask. Fill those pixels with
    # the nearest observed paper colour so the old background cannot form a halo.
    outside = ~cropped_mask
    nearest = ndimage.distance_transform_edt(
        outside, return_distances=False, return_indices=True)
    cropped_image[outside] = cropped_image[
        nearest[0][outside], nearest[1][outside]
    ]
    return Image.fromarray(np.dstack((
        cropped_image, cropped_mask.astype(np.uint8) * 255)), mode="RGBA")


def piece_from_page(page_rgb: np.ndarray, piece_mask: np.ndarray) -> Image.Image:
    if page_rgb.shape[:2] != piece_mask.shape:
        raise ValueError("Page RGB and piece mask shapes differ")
    ys, xs = np.nonzero(piece_mask)
    if len(xs) < 50:
        raise ValueError("Synthetic piece is empty or too small")
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    rgb = page_rgb[top:bottom, left:right].copy()
    mask = piece_mask[top:bottom, left:right]
    return Image.fromarray(np.dstack((rgb, mask.astype(np.uint8) * 255)), mode="RGBA")


def retexture_real_shape(piece: Image.Image, page_rgb: np.ndarray,
                         rng: np.random.Generator) -> Image.Image:
    """Keep a real contour but replace its interior with another PDF page crop."""
    piece = piece.convert("RGBA")
    mask = np.asarray(piece.getchannel("A"), dtype=np.uint8) >= 128
    height, width = mask.shape
    source = Image.fromarray(page_rgb, mode="RGB")
    scale = max(width / source.width, height / source.height) * float(rng.uniform(1.0, 1.55))
    resized = source.resize((max(width, round(source.width * scale)),
                             max(height, round(source.height * scale))),
                            Image.Resampling.LANCZOS)
    x = int(rng.integers(0, resized.width - width + 1))
    y = int(rng.integers(0, resized.height - height + 1))
    texture = np.asarray(resized.crop((x, y, x + width, y + height)), dtype=np.float32)
    paper_tint = rng.uniform(0.94, 1.06, 3).astype(np.float32)
    texture = np.clip(texture * paper_tint[None, None, :], 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack((texture, mask.astype(np.uint8) * 255)), mode="RGBA")


def _perspective_coefficients(destination: list[tuple[float, float]],
                              source: list[tuple[float, float]]) -> tuple[float, ...]:
    matrix = []
    values = []
    for (dx, dy), (sx, sy) in zip(destination, source):
        matrix.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        matrix.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
        values.extend((sx, sy))
    solution = np.linalg.solve(np.asarray(matrix, dtype=np.float64),
                               np.asarray(values, dtype=np.float64))
    return tuple(float(value) for value in solution)


def _perspective_warp(piece: Image.Image, mask: Image.Image,
                      rng: np.random.Generator) -> tuple[Image.Image, Image.Image, float]:
    width, height = piece.size
    strength = float(rng.uniform(0.015, 0.115))
    maximum = strength * min(width, height)
    destination = [
        (float(rng.uniform(0, maximum)), float(rng.uniform(0, maximum))),
        (width - 1 - float(rng.uniform(0, maximum)), float(rng.uniform(0, maximum))),
        (width - 1 - float(rng.uniform(0, maximum)),
         height - 1 - float(rng.uniform(0, maximum))),
        (float(rng.uniform(0, maximum)),
         height - 1 - float(rng.uniform(0, maximum))),
    ]
    source = [(0.0, 0.0), (width - 1.0, 0.0),
              (width - 1.0, height - 1.0), (0.0, height - 1.0)]
    try:
        coefficients = _perspective_coefficients(destination, source)
    except np.linalg.LinAlgError as error:
        raise CaptureGeometryError("Perspective transform is singular") from error
    transformed_piece = piece.transform(
        piece.size, Image.Transform.PERSPECTIVE, coefficients,
        resample=Image.Resampling.BICUBIC,
    )
    transformed_mask = mask.transform(
        mask.size, Image.Transform.PERSPECTIVE, coefficients,
        resample=Image.Resampling.NEAREST,
    )
    return transformed_piece, transformed_mask, strength


def _transform_piece(piece_rgba: Image.Image, output_size: int, profile: str,
                     rng: np.random.Generator
                     ) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    piece_rgba = piece_rgba.convert("RGBA")
    hard_mask = piece_rgba.getchannel("A").point(lambda value: 255 if value >= 128 else 0)
    piece_rgba.putalpha(hard_mask)
    perspective_strength = 0.0
    if profile == "phone":
        piece_rgba, hard_mask, perspective_strength = _perspective_warp(
            piece_rgba, hard_mask, rng)
    angle = float(rng.uniform(0.0, 360.0))
    piece_rgba = piece_rgba.rotate(angle, Image.Resampling.BICUBIC, expand=True)
    hard_mask = hard_mask.rotate(angle, Image.Resampling.NEAREST, expand=True)

    if rng.random() < 0.16:
        long_side_fraction = float(rng.uniform(0.13, 0.30))
    else:
        long_side_fraction = float(rng.uniform(0.30, 0.82))
    scale = output_size * long_side_fraction / max(piece_rgba.size)
    size = (max(3, round(piece_rgba.width * scale)),
            max(3, round(piece_rgba.height * scale)))
    piece_rgba = piece_rgba.resize(size, Image.Resampling.LANCZOS)
    hard_mask = hard_mask.resize(size, Image.Resampling.NEAREST)
    hard_mask = hard_mask.point(lambda value: 255 if value >= 128 else 0)

    bbox = hard_mask.getbbox()
    if bbox is None:
        raise CaptureGeometryError(
            "Piece disappeared during geometric transformation")
    piece_rgba = piece_rgba.crop(bbox)
    hard_mask = hard_mask.crop(bbox)
    # Rendering alpha may be soft, but truth remains hard and is saved separately.
    soft_alpha = hard_mask.filter(ImageFilter.GaussianBlur(radius=0.45))
    piece_rgba.putalpha(soft_alpha)
    return piece_rgba, hard_mask, {
        "rotation_degrees": angle,
        "long_side_fraction": long_side_fraction,
        "perspective_strength": perspective_strength,
    }


def _load_background(path: Path | None, size: int,
                     rng: np.random.Generator) -> np.ndarray:
    if path is None:
        anchors = np.asarray([
            (12, 13, 16), (42, 45, 50), (96, 88, 78), (175, 165, 150),
            (225, 225, 218), (33, 80, 142), (42, 125, 90), (120, 68, 54),
        ], dtype=np.float32)
        base = anchors[int(rng.integers(0, len(anchors)))]
        coarse = rng.normal(0.0, 1.0, (6, 6)).astype(np.float32)
        field = np.array(Image.fromarray(coarse, mode="F").resize(
            (size, size), Image.Resampling.BICUBIC), dtype=np.float32, copy=True)
        field -= float(field.mean())
        field /= max(float(field.std()), 1.0e-6)
        image = base[None, None, :] + field[:, :, None] * float(rng.uniform(2.0, 15.0))
        image += rng.normal(0.0, float(rng.uniform(0.4, 3.0)), image.shape)
        return np.uint8(np.clip(image, 0, 255))
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source).convert("RGB")
        zoom = float(rng.uniform(1.0, 1.35))
        scale = max(size / source.width, size / source.height) * zoom
        resized = source.resize((max(size, round(source.width * scale)),
                                 max(size, round(source.height * scale))),
                                Image.Resampling.LANCZOS)
        left = int(rng.integers(0, resized.width - size + 1))
        top = int(rng.integers(0, resized.height - size + 1))
        return np.asarray(resized.crop((left, top, left + size, top + size)),
                          dtype=np.uint8)


def _add_shadow(canvas: Image.Image, mask: Image.Image, profile: str,
                rng: np.random.Generator) -> tuple[Image.Image, dict[str, float]]:
    size = canvas.width
    full_mask = mask.convert("L")
    if profile == "phone":
        distance = float(rng.uniform(0.004, 0.035)) * size
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        offset_x, offset_y = int(round(math_cos(angle) * distance)), int(round(math_sin(angle) * distance))
        blur = float(rng.uniform(0.006, 0.026)) * size
        opacity = int(rng.integers(35, 125))
    else:
        offset_x, offset_y = int(rng.integers(-2, 4)), int(rng.integers(1, 6))
        blur = float(rng.uniform(0.002, 0.010)) * size
        opacity = int(rng.integers(20, 80))

    shifted = Image.new("L", canvas.size, 0)
    shifted.paste(full_mask, (offset_x, offset_y))
    shifted = shifted.filter(ImageFilter.GaussianBlur(radius=max(0.8, blur)))
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow.putalpha(shifted.point(lambda value: value * opacity // 255))
    canvas = Image.alpha_composite(canvas, shadow)

    # A tight contact shadow prevents the floating-sticker appearance.
    contact = full_mask.filter(ImageFilter.GaussianBlur(radius=max(0.6, size * 0.0015)))
    contact_shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    contact_shadow.putalpha(contact.point(lambda value: value * 28 // 255))
    canvas = Image.alpha_composite(canvas, contact_shadow)
    return canvas, {"shadow_offset_x": offset_x, "shadow_offset_y": offset_y,
                    "shadow_blur": blur, "shadow_opacity": opacity}


def math_cos(value: float) -> float:
    return float(np.cos(value))


def math_sin(value: float) -> float:
    return float(np.sin(value))


def _add_paper_reflection(image: np.ndarray, mask: np.ndarray,
                          rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    if not mask.any() or rng.random() > 0.42:
        return image, {"reflection_applied": False}
    height, width = mask.shape
    layer = Image.new("L", (width, height), 0)
    painter = ImageDraw.Draw(layer)
    center_x = int(rng.integers(-width // 5, width + width // 5))
    center_y = int(rng.integers(-height // 5, height + height // 5))
    radius_x = int(rng.uniform(0.10, 0.42) * width)
    radius_y = int(rng.uniform(0.035, 0.20) * height)
    painter.ellipse((center_x - radius_x, center_y - radius_y,
                     center_x + radius_x, center_y + radius_y), fill=255)
    layer = layer.rotate(float(rng.uniform(0.0, 180.0)),
                         Image.Resampling.BICUBIC, center=(center_x, center_y))
    layer = layer.filter(ImageFilter.GaussianBlur(
        radius=float(rng.uniform(0.025, 0.085)) * min(width, height)))
    strength = float(rng.uniform(0.08, 0.38))
    alpha = np.asarray(layer, dtype=np.float32) / 255.0 * strength
    alpha *= mask.astype(np.float32)
    tint = rng.uniform(0.92, 1.08, 3).astype(np.float32)
    result = image.astype(np.float32) / 255.0
    # Screen blend approximates a broad reflected light source without erasing texture.
    reflected = 1.0 - (1.0 - result) * (1.0 - tint[None, None, :] * alpha[:, :, None])
    return np.uint8(np.clip(reflected, 0.0, 1.0) * 255.0), {
        "reflection_applied": True,
        "reflection_strength": strength,
    }


def _radial_distort(image: np.ndarray, mask: np.ndarray, coefficient: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    scale = max(center_x, center_y, 1.0)
    dx, dy = (xx - center_x) / scale, (yy - center_y) / scale
    radius_squared = dx * dx + dy * dy
    source_x = center_x + dx * (1.0 + coefficient * radius_squared) * scale
    source_y = center_y + dy * (1.0 + coefficient * radius_squared) * scale
    distorted = np.empty_like(image)
    for channel in range(3):
        distorted[:, :, channel] = ndimage.map_coordinates(
            image[:, :, channel], (source_y, source_x), order=1, mode="reflect")
    distorted_mask = ndimage.map_coordinates(
        mask.astype(np.uint8), (source_y, source_x), order=0,
        mode="constant", cval=0).astype(bool)
    return distorted, distorted_mask


def _postprocess(image: np.ndarray, mask: np.ndarray, profile: str,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = mask.shape
    result = image.astype(np.float32) / 255.0
    y = np.linspace(-0.5, 0.5, height, dtype=np.float32)[:, None]
    x = np.linspace(-0.5, 0.5, width, dtype=np.float32)[None, :]
    if profile == "phone":
        gradient_x = float(rng.uniform(-0.30, 0.30))
        gradient_y = float(rng.uniform(-0.30, 0.30))
        vignette_strength = float(rng.uniform(0.0, 0.32))
        vignette = 1.0 - vignette_strength * np.clip((x * x + y * y) / 0.5, 0.0, 1.0)
        illumination = (1.0 + gradient_x * x + gradient_y * y) * vignette
    else:
        gradient_x = float(rng.uniform(-0.10, 0.10))
        gradient_y = float(rng.uniform(-0.18, 0.18))
        vignette_strength = 0.0
        frequency = float(rng.uniform(2.0, 14.0))
        bands = np.sin(np.linspace(0.0, 2.0 * np.pi * frequency, height,
                                   dtype=np.float32))[:, None]
        illumination = 1.0 + gradient_x * x + gradient_y * y
        illumination += bands * float(rng.uniform(0.0, 0.035))
    result *= illumination[:, :, None]
    colour = rng.normal(1.0, 0.035 if profile == "phone" else 0.018, 3)
    result *= colour[None, None, :]
    result = np.clip(result, 0.0, 1.0)

    contrast = float(rng.uniform(0.82, 1.20) if profile == "phone"
                     else rng.uniform(0.90, 1.12))
    gamma = float(rng.uniform(0.78, 1.28) if profile == "phone"
                  else rng.uniform(0.90, 1.12))
    result = np.clip(0.5 + (result - 0.5) * contrast, 0.0, 1.0)
    result = np.power(result, gamma)
    output = np.uint8(np.clip(result, 0.0, 1.0) * 255.0)

    radial_coefficient = 0.0
    if profile == "phone" and rng.random() < 0.38:
        radial_coefficient = float(rng.uniform(-0.055, 0.045))
        output, mask = _radial_distort(output, mask, radial_coefficient)

    blur_radius = 0.0
    if rng.random() < (0.36 if profile == "phone" else 0.16):
        blur_radius = float(rng.uniform(0.15, 1.35 if profile == "phone" else 0.75))
        output = np.asarray(Image.fromarray(output).filter(
            ImageFilter.GaussianBlur(radius=blur_radius)), dtype=np.uint8)
    noise_std = float(rng.uniform(0.2, 5.0) if profile == "phone"
                      else rng.uniform(0.1, 2.5))
    output = np.uint8(np.clip(
        output.astype(np.float32) + rng.normal(0.0, noise_std, output.shape),
        0, 255))

    jpeg_quality = None
    if rng.random() < (0.62 if profile == "phone" else 0.20):
        jpeg_quality = int(rng.integers(68 if profile == "phone" else 82, 99))
        stream = io.BytesIO()
        Image.fromarray(output).save(stream, format="JPEG", quality=jpeg_quality,
                                     subsampling=0)
        stream.seek(0)
        with Image.open(stream) as decoded:
            output = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
    return output, mask, {
        "illumination_gradient_x": gradient_x,
        "illumination_gradient_y": gradient_y,
        "vignette_strength": vignette_strength,
        "contrast": contrast,
        "gamma": gamma,
        "radial_distortion": radial_coefficient,
        "blur_radius": blur_radius,
        "noise_std_255": noise_std,
        "jpeg_quality": jpeg_quality,
    }


def _render_capture_once(piece_rgba: Image.Image, background_path: Path | None,
                         profile: str, output_size: int, seed: int
                         ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if profile not in {"phone", "scanner"}:
        raise ValueError("profile must be phone or scanner")
    if output_size < 128:
        raise ValueError("output_size must be at least 128")
    rng = np.random.default_rng(int(seed))
    background = _load_background(background_path, output_size, rng)
    canvas = Image.fromarray(background, mode="RGB").convert("RGBA")
    piece, local_mask, transform = _transform_piece(
        piece_rgba, output_size, profile, rng)
    margin = max(3, round(output_size * 0.025))
    if piece.width > output_size - 2 * margin or piece.height > output_size - 2 * margin:
        raise CaptureGeometryError(
            "Transformed piece is too large for the output canvas")
    left = int(rng.integers(margin, output_size - piece.width - margin + 1))
    top = int(rng.integers(margin, output_size - piece.height - margin + 1))
    full_mask_image = Image.new("L", (output_size, output_size), 0)
    full_mask_image.paste(local_mask, (left, top))
    canvas, shadow = _add_shadow(canvas, full_mask_image, profile, rng)
    canvas.alpha_composite(piece, (left, top))
    mask = np.asarray(full_mask_image, dtype=np.uint8) >= 128
    image = np.asarray(canvas.convert("RGB"), dtype=np.uint8)
    if profile == "phone":
        image, reflection = _add_paper_reflection(image, mask, rng)
    else:
        reflection = {"reflection_applied": False}
    image, mask, post = _postprocess(image, mask, profile, rng)
    return image, mask, {
        "seed": int(seed),
        "capture_profile": profile,
        "background_path": None if background_path is None else str(background_path),
        "foreground_fraction": float(mask.mean()),
        **transform,
        **shadow,
        **reflection,
        **post,
    }


def render_capture(piece_rgba: Image.Image, background_path: Path | None,
                   profile: str, output_size: int, seed: int,
                   maximum_attempts: int = 10
                   ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Render deterministically, retrying poses that break a thin raster neck.

    A retry changes the sampled scale/pose instead of altering the target mask
    with morphology. This preserves the physical contour represented by the
    transformed source while guaranteeing one connected foreground component.
    """
    if maximum_attempts < 1:
        raise ValueError("maximum_attempts must be positive")
    last_problem = "no attempts"
    for attempt in range(maximum_attempts):
        trial_seed = int(seed) if attempt == 0 else (
            int(seed) ^ ((attempt + 1) * 0x5DEECE66D)) & 0x7FFF_FFFF_FFFF_FFFF
        try:
            image, mask, metadata = _render_capture_once(
                piece_rgba, background_path, profile, output_size, trial_seed)
        except CaptureGeometryError as error:
            last_problem = str(error)
            continue
        component_count = ndimage.label(
            mask, np.ones((3, 3), dtype=np.uint8))[1]
        foreground_pixels = int(mask.sum())
        if foreground_pixels >= 50 and component_count == 1:
            metadata["requested_seed"] = int(seed)
            metadata["render_attempt"] = attempt + 1
            return image, mask, metadata
        last_problem = (
            f"foreground_pixels={foreground_pixels}, components={component_count}")
    raise CaptureRenderError(
        f"Could not render one connected paper mask after {maximum_attempts} attempts: "
        f"{last_problem}")


def render_background_only(background_path: Path | None, profile: str,
                           output_size: int, seed: int
                           ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    image = _load_background(background_path, output_size, rng)
    mask = np.zeros((output_size, output_size), dtype=bool)
    image, mask, metadata = _postprocess(image, mask, profile, rng)
    return image, mask, {
        "seed": int(seed),
        "capture_profile": profile,
        "background_path": None if background_path is None else str(background_path),
        "foreground_fraction": 0.0,
        **metadata,
    }
