"""Vector-first irregular page partitions for synthetic paper fragments.

The sequential family models a page being cut or torn one current fragment at
a time.  Legacy-style global families are retained as minority distributions
for controlled topology diversity.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import polygonize, split, unary_union


class GeometryError(ValueError):
    pass


@dataclass
class Partition:
    pieces: list[Polygon]
    paths: list[LineString]
    cut_styles: list[str]
    family: str
    attempts: int
    rejections: dict[str, int]


def _normalised_noise(count: int, anchors: int,
                      rng: np.random.Generator) -> np.ndarray:
    values = rng.normal(0.0, 1.0, anchors)
    values[0] = values[-1] = 0.0
    curve = CubicSpline(np.linspace(0.0, 1.0, anchors), values,
                        bc_type="natural")(np.linspace(0.0, 1.0, count))
    scale = max(1.0e-6, float(np.max(np.abs(curve))))
    return curve / scale


def _irregular_path(start: tuple[float, float], end: tuple[float, float],
                    scale: float, rng: np.random.Generator,
                    style: str) -> LineString:
    first = np.asarray(start, dtype=np.float64)
    last = np.asarray(end, dtype=np.float64)
    chord = last - first
    length = float(np.linalg.norm(chord))
    if length < max(12.0, scale * 0.12):
        raise GeometryError("cut endpoints are too close")
    normal = np.asarray([-chord[1], chord[0]], dtype=np.float64) / length
    count = max(48, int(math.ceil(length / 4.0)))
    t = np.linspace(0.0, 1.0, count)

    if style == "smooth_scissor":
        control_count = int(rng.integers(4, 8))
        offsets = rng.normal(0.0, 1.0, control_count)
        offsets[0] = offsets[-1] = 0.0
        drift = CubicSpline(np.linspace(0.0, 1.0, control_count), offsets,
                            bc_type="natural")(t)
        drift /= max(1.0, float(np.max(np.abs(drift))))
        drift *= float(rng.uniform(0.025, 0.14)) * scale
        drift += (_normalised_noise(count, 21, rng) *
                  float(rng.uniform(0.0004, 0.0025)) * scale)
    elif style == "angular_scissor":
        knot_count = int(rng.integers(5, 11))
        knots = np.linspace(0.0, 1.0, knot_count)
        offsets = np.zeros(knot_count, dtype=np.float64)
        step = float(rng.uniform(0.018, 0.065)) * scale
        for index in range(1, knot_count - 1):
            offsets[index] = np.clip(
                offsets[index - 1] + rng.normal(0.0, step),
                -0.17 * scale,
                0.17 * scale,
            )
        drift = np.interp(t, knots, offsets)
    elif style == "rough_tear":
        broad = _normalised_noise(count, int(rng.integers(5, 9)), rng)
        medium = _normalised_noise(count, int(rng.integers(17, 31)), rng)
        fine = _normalised_noise(count, int(rng.integers(43, 75)), rng)
        drift = (
            broad * float(rng.uniform(0.035, 0.15)) * scale
            + medium * float(rng.uniform(0.004, 0.018)) * scale
            + fine * float(rng.uniform(0.0008, 0.004)) * scale
        )
        drift *= np.sin(np.pi * t) ** 0.45
    elif style == "hybrid":
        broad = _normalised_noise(count, int(rng.integers(5, 9)), rng)
        angular_knots = np.linspace(0.0, 1.0, int(rng.integers(5, 9)))
        angular_values = rng.normal(0.0, 0.05 * scale, len(angular_knots))
        angular_values[0] = angular_values[-1] = 0.0
        drift = (broad * float(rng.uniform(0.025, 0.10)) * scale
                 + np.interp(t, angular_knots, angular_values))
        drift *= np.sin(np.pi * t) ** 0.6
    elif style == "straight_imperfect":
        drift = (_normalised_noise(count, int(rng.integers(9, 19)), rng) *
                 float(rng.uniform(0.001, 0.012)) * scale * np.sin(np.pi * t))
    else:
        raise GeometryError(f"unknown cut style: {style}")

    points = first[None, :] + t[:, None] * chord[None, :] + drift[:, None] * normal[None, :]
    points[0], points[-1] = first, last
    path = LineString(points)
    if not path.is_simple:
        raise GeometryError("generated cut self-intersects")
    return path


def _sample_boundary_pair(polygon: Polygon, rng: np.random.Generator
                          ) -> tuple[tuple[float, float], tuple[float, float]]:
    boundary = polygon.exterior
    perimeter = boundary.length
    first_distance = float(rng.uniform(0.0, perimeter))
    # A wide distribution allows both opposite-side cuts and same-side notches.
    separation = float(rng.uniform(0.16, 0.58)) * perimeter
    if rng.random() < 0.5:
        separation = -separation
    second_distance = (first_distance + separation) % perimeter
    first = boundary.interpolate(first_distance)
    second = boundary.interpolate(second_distance)
    return (float(first.x), float(first.y)), (float(second.x), float(second.y))


def _valid_split(polygon: Polygon, path: LineString, page_area: float,
                 min_page_area_ratio: float) -> list[Polygon]:
    if not polygon.buffer(1.0e-6).covers(path):
        raise GeometryError("cut leaves the selected fragment")
    result = split(polygon, path)
    pieces = [part for part in result.geoms if isinstance(part, Polygon)]
    if len(pieces) != 2:
        raise GeometryError("cut did not divide one fragment into exactly two")
    for piece in pieces:
        if not piece.is_valid or piece.interiors:
            raise GeometryError("cut produced invalid geometry or a hole")
        if piece.area < page_area * min_page_area_ratio:
            raise GeometryError("cut produced a tiny fragment")
    return pieces


def generate_sequential_partition(width: int, height: int, target_pieces: int,
                                  rng: np.random.Generator,
                                  max_attempts_per_cut: int = 160,
                                  min_piece_area_ratio: float = 0.008) -> Partition:
    if width < 80 or height < 80 or target_pieces < 2:
        raise GeometryError("invalid page dimensions or target piece count")
    page = box(0.0, 0.0, float(width), float(height))
    page_area = page.area
    pieces: list[Polygon] = [page]
    paths: list[LineString] = []
    styles: list[str] = []
    rejections: Counter[str] = Counter()
    total_attempts = 0
    style_names = (
        "smooth_scissor", "angular_scissor", "rough_tear", "hybrid",
        "straight_imperfect",
    )
    style_probabilities = np.asarray([0.28, 0.23, 0.25, 0.17, 0.07])

    while len(pieces) < target_pieces:
        areas = np.asarray([piece.area for piece in pieces], dtype=np.float64)
        eligible = areas >= page_area * min_piece_area_ratio * 2.4
        if not eligible.any():
            raise GeometryError("no fragment is large enough for another valid cut")
        weights = np.where(eligible, areas ** 1.25, 0.0)
        weights /= weights.sum()
        selected_index = int(rng.choice(len(pieces), p=weights))
        selected = pieces[selected_index]
        min_x, min_y, max_x, max_y = selected.bounds
        scale = min(max_x - min_x, max_y - min_y)

        accepted = False
        for _ in range(max_attempts_per_cut):
            total_attempts += 1
            style = str(rng.choice(style_names, p=style_probabilities))
            try:
                start, end = _sample_boundary_pair(selected, rng)
                path = _irregular_path(start, end, scale, rng, style)
                new_pieces = _valid_split(
                    selected, path, page_area, min_piece_area_ratio)
            except GeometryError as error:
                rejections[str(error)] += 1
                continue
            pieces[selected_index:selected_index + 1] = new_pieces
            paths.append(path)
            styles.append(style)
            accepted = True
            break
        if not accepted:
            raise GeometryError(
                f"could not split fragment {len(pieces)} after {max_attempts_per_cut} attempts"
            )

    pieces.sort(key=lambda polygon: (round(polygon.centroid.y, 6),
                                     round(polygon.centroid.x, 6)))
    _validate_partition(pieces, width, height, min_piece_area_ratio)
    return Partition(pieces, paths, styles, "sequential", total_attempts,
                     dict(rejections))


def _rectangle_chord(width: int, height: int, angle: float, offset: float
                     ) -> tuple[tuple[float, float], tuple[float, float]]:
    direction = np.asarray([math.cos(angle), math.sin(angle)])
    normal = np.asarray([-direction[1], direction[0]])
    center = np.asarray([width / 2.0, height / 2.0]) + offset * normal
    candidates: list[tuple[float, tuple[float, float]]] = []
    for axis, maximum in ((0, width), (1, height)):
        if abs(direction[axis]) < 1.0e-9:
            continue
        for value in (0.0, float(maximum)):
            distance = (value - center[axis]) / direction[axis]
            point = center + distance * direction
            other_maximum = height if axis == 0 else width
            if 1.0e-4 < point[1 - axis] < other_maximum - 1.0e-4:
                point[axis] = value
                candidates.append((float(distance), (float(point[0]), float(point[1]))))
    if len(candidates) != 2:
        raise GeometryError("global line misses the page or meets a corner")
    candidates.sort()
    return candidates[0][1], candidates[1][1]


def _partition_from_global_paths(paths: list[LineString], width: int, height: int,
                                 target_pieces: int,
                                 min_piece_area_ratio: float) -> list[Polygon]:
    page = box(0.0, 0.0, float(width), float(height))
    for path in paths:
        endpoints = MultiPoint([Point(path.coords[0]), Point(path.coords[-1])])
        if not path.is_simple or not page.covers(path):
            raise GeometryError("invalid global cut")
        if not path.intersection(page.boundary).equals(endpoints):
            raise GeometryError("global cut touches the border away from its endpoints")
    faces = [polygon for polygon in polygonize(unary_union([page.boundary, *paths]))]
    if len(faces) != target_pieces:
        raise GeometryError("global paths produced the wrong number of pieces")
    _validate_partition(faces, width, height, min_piece_area_ratio)
    return faces


def generate_global_partition(width: int, height: int, target_pieces: int,
                              rng: np.random.Generator, family: str,
                              retries: int = 180,
                              min_piece_area_ratio: float = 0.008) -> Partition:
    if family not in {"parallel", "crossing", "global_freeform"}:
        raise GeometryError(f"unknown global family: {family}")
    rejections: Counter[str] = Counter()
    for attempt in range(1, retries + 1):
        try:
            if family == "parallel":
                groups = (target_pieces - 1, 0)
            elif family == "crossing":
                factor_pairs = [
                    (rows - 1, target_pieces // rows - 1)
                    for rows in range(2, target_pieces + 1)
                    if target_pieces % rows == 0 and target_pieces // rows >= 2
                ]
                if not factor_pairs:
                    raise GeometryError("crossing family requires a composite piece count")
                groups = factor_pairs[int(rng.integers(0, len(factor_pairs)))]
            else:
                groups = (0, 0)

            paths: list[LineString] = []
            styles: list[str] = []
            if family in {"parallel", "crossing"}:
                base_angle = float(rng.uniform(0.0, 2.0 * np.pi))
                for group_index, amount in enumerate(groups):
                    angle = base_angle + group_index * (
                        np.pi / 2.0 + float(rng.uniform(-0.11, 0.11)))
                    extent = (abs(width * math.sin(angle)) / 2.0
                              + abs(height * math.cos(angle)) / 2.0)
                    for index in range(amount):
                        offset = extent * (2.0 * (index + 1) / (amount + 1) - 1.0)
                        offset += float(rng.uniform(-0.025, 0.025)) * min(width, height)
                        start, end = _rectangle_chord(width, height, angle, offset)
                        style = str(rng.choice(("smooth_scissor", "rough_tear", "hybrid")))
                        paths.append(_irregular_path(start, end, min(width, height), rng, style))
                        styles.append(style)
            else:
                # Add arbitrary border-to-border paths only when they increase
                # the number of valid faces without overshooting the target.
                current_count = 1
                for _ in range(target_pieces + 3):
                    found = False
                    for _inner in range(80):
                        side_a = int(rng.integers(0, 4))
                        side_b = (side_a if rng.random() < 0.18 else
                                  int(rng.choice([side for side in range(4) if side != side_a])))
                        def border_point(side: int) -> tuple[float, float]:
                            value = float(rng.uniform(0.08, 0.92))
                            return ((0.0, value * height), (float(width), value * height),
                                    (value * width, 0.0), (value * width, float(height)))[side]
                        style = str(rng.choice(("smooth_scissor", "angular_scissor",
                                                "rough_tear", "hybrid")))
                        candidate = _irregular_path(border_point(side_a), border_point(side_b),
                                                    min(width, height), rng, style)
                        try:
                            tentative = list(polygonize(unary_union([
                                box(0.0, 0.0, float(width), float(height)).boundary,
                                *paths, candidate,
                            ])))
                        except Exception as error:  # GEOS can reject degenerate candidates.
                            raise GeometryError("global polygonization failed") from error
                        if current_count < len(tentative) <= target_pieces:
                            paths.append(candidate)
                            styles.append(style)
                            current_count = len(tentative)
                            found = True
                            break
                    if current_count == target_pieces:
                        break
                    if not found:
                        raise GeometryError("freeform global sampler stalled")
            pieces = _partition_from_global_paths(
                paths, width, height, target_pieces, min_piece_area_ratio)
            pieces.sort(key=lambda polygon: (round(polygon.centroid.y, 6),
                                             round(polygon.centroid.x, 6)))
            return Partition(pieces, paths, styles, family, attempt, dict(rejections))
        except GeometryError as error:
            rejections[str(error)] += 1
    raise GeometryError(f"could not generate {family} partition after {retries} attempts")


def generate_partition(width: int, height: int, target_pieces: int,
                       rng: np.random.Generator, family: str = "auto") -> Partition:
    if family == "auto":
        family = str(rng.choice(
            ("sequential", "global_freeform", "crossing", "parallel"),
            p=(0.60, 0.20, 0.10, 0.10),
        ))
    if family == "sequential":
        return generate_sequential_partition(width, height, target_pieces, rng)
    return generate_global_partition(width, height, target_pieces, rng, family)


def _validate_partition(pieces: list[Polygon], width: int, height: int,
                        min_piece_area_ratio: float) -> None:
    page = box(0.0, 0.0, float(width), float(height))
    if not pieces:
        raise GeometryError("partition has no pieces")
    if any(not piece.is_valid or piece.interiors for piece in pieces):
        raise GeometryError("partition contains invalid pieces or holes")
    if any(piece.area < page.area * min_piece_area_ratio for piece in pieces):
        raise GeometryError("partition contains a tiny piece")
    union = unary_union(pieces)
    if page.symmetric_difference(union).area > page.area * 1.0e-7:
        raise GeometryError("partition does not exactly cover the source page")
    overlap = sum(piece.area for piece in pieces) - union.area
    if overlap > page.area * 1.0e-8:
        raise GeometryError("partition pieces overlap")


def rasterize_partition(partition: Partition, width: int, height: int) -> np.ndarray:
    """Rasterize complementary labels while assigning every page pixel once."""
    from PIL import Image, ImageDraw
    from scipy import ndimage

    labels = np.zeros((height, width), dtype=np.uint16)
    for index, polygon in sorted(enumerate(partition.pieces),
                                 key=lambda pair: -pair[1].area):
        canvas = Image.new("L", (width, height), 0)
        painter = ImageDraw.Draw(canvas)
        painter.polygon(list(polygon.exterior.coords), fill=255)
        mask = np.asarray(canvas, dtype=np.uint8) > 0
        labels[mask] = index + 1
    missing = labels == 0
    if missing.any():
        nearest = ndimage.distance_transform_edt(
            missing, return_distances=False, return_indices=True)
        labels[missing] = labels[tuple(axis[missing] for axis in nearest)]
    for index in range(len(partition.pieces)):
        mask = labels == index + 1
        if mask.sum() < 50 or ndimage.label(mask, np.ones((3, 3), dtype=np.uint8))[1] != 1:
            raise GeometryError("rasterization produced an empty or disconnected piece")
    return labels
