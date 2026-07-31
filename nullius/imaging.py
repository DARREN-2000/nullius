"""Lesion image preprocessing and deterministic feature extraction, pure Python.

The design rule here is the same one that governs the text path: the model may
only ever see quantities a human can name and check. So preprocessing produces
nine named, unit-documented morphological features rather than a learned
embedding, and the classifier consumes only those. That buys three things a
black-box CNN does not give you:

  * Explanations that are true by construction - the attribution is over
    "border irregularity", not over pixel 4,182.
  * An out-of-distribution check that means something, because the feature
    space is low-dimensional and its training statistics are known.
  * Reviewability: every number below can be recomputed by hand from the mask.

The cost is stated in ADR-009: hand-designed features leave accuracy on the
table versus a trained CNN on real data. On synthetic data that trade is free,
which is precisely why the accuracy numbers here are not a clinical claim.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass, field
from typing import Sequence

from .dicom import Dataset

Image = list[list[float]]
Mask = list[list[bool]]

WORKING_SIZE = 128

# Order is part of the model contract: the ONNX graph consumes this exact order.
FEATURE_NAMES: tuple[str, ...] = (
    "area_fraction",
    "asymmetry",
    "border_irregularity",
    "edge_gradient",
    "contrast",
    "intensity_variance",
    "variegation",
    "eccentricity",
    "diameter_norm",
)

FEATURE_LABELS: dict[str, str] = {
    "area_fraction": "Lesion area as a share of the field",
    "asymmetry": "Asymmetry across the principal axes (ABCD: A)",
    "border_irregularity": "Border irregularity, normalised isoperimetric deficit (ABCD: B)",
    "edge_gradient": "Border sharpness, mean gradient across the boundary",
    "contrast": "Lesion-to-skin intensity contrast",
    "intensity_variance": "Internal intensity variance",
    "variegation": "Internal variegation, 10th-90th percentile spread (ABCD: C)",
    "eccentricity": "Elongation from second-order moments",
    "diameter_norm": "Maximum diameter in mm, normalised at 10 mm (ABCD: D)",
}


@dataclass
class Preprocessed:
    image: Image
    mask: Mask
    features: dict[str, float]
    quality: dict[str, float]
    diameter_mm: float
    steps: list[str] = field(default_factory=list)

    def vector(self) -> list[float]:
        return [self.features[name] for name in FEATURE_NAMES]


# ------------------------------------------------------------------ basic ops
def from_dataset(dataset: Dataset) -> Image:
    """Apply the stored window/level and normalise to 0..1."""
    if dataset.get("PhotometricInterpretation", "MONOCHROME2") != "MONOCHROME2":
        raise ValueError("only MONOCHROME2 is supported")
    pixels = dataset.pixels()
    rows, cols = dataset.rows, dataset.columns
    centre = float(dataset.get("WindowCenter", 2048))
    width = max(float(dataset.get("WindowWidth", 4096)), 1.0)
    low = centre - width / 2.0
    return [
        [min(1.0, max(0.0, (pixels[r * cols + c] - low) / width)) for c in range(cols)]
        for r in range(rows)
    ]


def median_filter(image: Image, radius: int = 1) -> Image:
    """Suppresses hair, speckle and single-pixel sensor noise before segmentation."""
    rows, cols = len(image), len(image[0])
    out = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        r0, r1 = max(0, r - radius), min(rows - 1, r + radius)
        for c in range(cols):
            c0, c1 = max(0, c - radius), min(cols - 1, c + radius)
            window = [image[rr][cc] for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)]
            window.sort()
            out[r][c] = window[len(window) // 2]
    return out


def resize(image: Image, size: int = WORKING_SIZE) -> Image:
    """Bilinear resample to a fixed working resolution."""
    rows, cols = len(image), len(image[0])
    if rows == size and cols == size:
        return [row[:] for row in image]
    out = [[0.0] * size for _ in range(size)]
    for r in range(size):
        sy = (r + 0.5) * rows / size - 0.5
        y0 = max(0, min(rows - 1, int(math.floor(sy))))
        y1 = min(rows - 1, y0 + 1)
        wy = sy - y0
        for c in range(size):
            sx = (c + 0.5) * cols / size - 0.5
            x0 = max(0, min(cols - 1, int(math.floor(sx))))
            x1 = min(cols - 1, x0 + 1)
            wx = sx - x0
            top = image[y0][x0] * (1 - wx) + image[y0][x1] * wx
            bottom = image[y1][x0] * (1 - wx) + image[y1][x1] * wx
            out[r][c] = top * (1 - wy) + bottom * wy
    return out


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def normalise_illumination(image: Image) -> Image:
    """Rescale between the 2nd and 98th percentile.

    Dermoscopy images vary hugely in exposure and vignetting between devices.
    Without this, brightness differences between scanners dominate the features
    and the classifier silently learns the device rather than the lesion.
    """
    flat = [v for row in image for v in row]
    low, high = _percentile(flat, 0.02), _percentile(flat, 0.98)
    span = max(high - low, 1e-6)
    return [[min(1.0, max(0.0, (v - low) / span)) for v in row] for row in image]


def otsu_threshold(image: Image, bins: int = 64) -> float:
    """Otsu's method: the threshold maximising between-class variance."""
    flat = [v for row in image for v in row]
    histogram = [0] * bins
    for value in flat:
        histogram[min(bins - 1, int(value * bins))] += 1
    total = len(flat)
    sum_all = sum(i * histogram[i] for i in range(bins))
    sum_b = 0.0
    weight_b = 0
    best_variance, best_threshold = -1.0, 0.5
    for i in range(bins):
        weight_b += histogram[i]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += i * histogram[i]
        mean_b = sum_b / weight_b
        mean_f = (sum_all - sum_b) / weight_f
        variance = weight_b * weight_f * (mean_b - mean_f) ** 2
        if variance > best_variance:
            best_variance, best_threshold = variance, (i + 0.5) / bins
    return best_threshold


def largest_component(mask: Mask) -> Mask:
    """Keep only the biggest 4-connected blob, dropping ink marks and rulers."""
    rows, cols = len(mask), len(mask[0])
    seen = [[False] * cols for _ in range(rows)]
    best: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r][c] or seen[r][c]:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            component: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                component.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr][nc] and not seen[nr][nc]:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            if len(component) > len(best):
                best = component
    out = [[False] * cols for _ in range(rows)]
    for r, c in best:
        out[r][c] = True
    return out


def erode(mask: Mask, iterations: int = 1) -> Mask:
    """Shrink a mask by its 4-connected boundary, `iterations` times.

    Interior texture must be measured away from the rim. Without this, a small
    lesion is mostly soft edge, and 'internal variegation' silently becomes a
    measurement of blur - which is how a feature ends up meaning the opposite
    of its name.
    """
    rows, cols = len(mask), len(mask[0])
    current = mask
    for _ in range(iterations):
        nxt = [[False] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if not current[r][c]:
                    continue
                if all(
                    0 <= nr < rows and 0 <= nc < cols and current[nr][nc]
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                ):
                    nxt[r][c] = True
        current = nxt
    return current


def segment(image: Image) -> Mask:
    """Lesion pixels are darker than surrounding skin, so threshold from below."""
    threshold = otsu_threshold(image)
    raw = [[value < threshold for value in row] for row in image]
    return largest_component(raw)


# ------------------------------------------------------------------ features
def _moments(mask: Mask) -> tuple[float, float, float, float, float, float]:
    rows, cols = len(mask), len(mask[0])
    area = 0
    sum_r = sum_c = 0.0
    for r in range(rows):
        for c in range(cols):
            if mask[r][c]:
                area += 1
                sum_r += r
                sum_c += c
    if area == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    cr, cc = sum_r / area, sum_c / area
    mu20 = mu02 = mu11 = 0.0
    for r in range(rows):
        for c in range(cols):
            if mask[r][c]:
                dr, dc = r - cr, c - cc
                mu20 += dc * dc
                mu02 += dr * dr
                mu11 += dr * dc
    return float(area), cr, cc, mu20 / area, mu02 / area, mu11 / area


def _perimeter(mask: Mask) -> int:
    rows, cols = len(mask), len(mask[0])
    count = 0
    for r in range(rows):
        for c in range(cols):
            if not mask[r][c]:
                continue
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= nr < rows and 0 <= nc < cols) or not mask[nr][nc]:
                    count += 1
                    break
    return count


def _asymmetry(mask: Mask, centre_r: float, centre_c: float) -> float:
    """Mean mismatch fraction when the mask is folded about its centroid axes."""
    rows, cols = len(mask), len(mask[0])
    area = sum(1 for r in range(rows) for c in range(cols) if mask[r][c])
    if area == 0:
        return 0.0
    mismatch_v = mismatch_h = 0
    for r in range(rows):
        mirror_r = int(round(2 * centre_r - r))
        for c in range(cols):
            mirror_c = int(round(2 * centre_c - c))
            here = mask[r][c]
            other_v = mask[r][mirror_c] if 0 <= mirror_c < cols else False
            other_h = mask[mirror_r][c] if 0 <= mirror_r < rows else False
            mismatch_v += here != other_v
            mismatch_h += here != other_h
    return min(1.0, (mismatch_v + mismatch_h) / (4.0 * area))


def extract_features(image: Image, mask: Mask, pixel_spacing_mm: float) -> tuple[dict[str, float], float]:
    rows, cols = len(image), len(image[0])
    area, centre_r, centre_c, mu20, mu02, mu11 = _moments(mask)
    total = float(rows * cols)
    if area == 0:
        empty = {name: 0.0 for name in FEATURE_NAMES}
        return empty, 0.0

    # Interior statistics are measured on an eroded mask so the softened rim
    # does not masquerade as internal texture.
    interior = erode(mask, 3)
    inside = [image[r][c] for r in range(rows) for c in range(cols) if interior[r][c]]
    if len(inside) < 12:
        inside = [image[r][c] for r in range(rows) for c in range(cols) if mask[r][c]]
    outside = [image[r][c] for r in range(rows) for c in range(cols) if not mask[r][c]]
    mean_in = sum(inside) / len(inside)
    mean_out = (sum(outside) / len(outside)) if outside else mean_in

    perimeter = _perimeter(mask)
    compactness = (perimeter * perimeter) / (4.0 * math.pi * area) if area else 1.0

    boundary_gradients: list[float] = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if not mask[r][c]:
                continue
            if mask[r - 1][c] and mask[r + 1][c] and mask[r][c - 1] and mask[r][c + 1]:
                continue
            gx = image[r][c + 1] - image[r][c - 1]
            gy = image[r + 1][c] - image[r - 1][c]
            boundary_gradients.append(math.hypot(gx, gy))
    edge_gradient = sum(boundary_gradients) / len(boundary_gradients) if boundary_gradients else 0.0

    variance = sum((v - mean_in) ** 2 for v in inside) / len(inside)
    variegation = _percentile(inside, 0.9) - _percentile(inside, 0.1)

    trace = mu20 + mu02
    diff = math.sqrt(max(0.0, (mu20 - mu02) ** 2 + 4 * mu11 * mu11))
    lam1, lam2 = (trace + diff) / 2.0, (trace - diff) / 2.0
    eccentricity = math.sqrt(max(0.0, 1.0 - (lam2 / lam1))) if lam1 > 1e-9 else 0.0

    max_extent = 0.0
    for r in range(rows):
        for c in range(cols):
            if mask[r][c]:
                max_extent = max(max_extent, 2.0 * math.hypot(r - centre_r, c - centre_c))
    diameter_mm = max_extent * pixel_spacing_mm

    features = {
        "area_fraction": area / total,
        "asymmetry": _asymmetry(mask, centre_r, centre_c),
        "border_irregularity": min(1.0, max(0.0, compactness - 1.0)),
        "edge_gradient": min(1.0, edge_gradient),
        "contrast": min(1.0, abs(mean_out - mean_in)),
        "intensity_variance": min(1.0, variance * 40.0),
        "variegation": min(1.0, variegation),
        "eccentricity": eccentricity,
        "diameter_norm": min(1.0, diameter_mm / 10.0),
    }
    return {k: round(v, 6) for k, v in features.items()}, round(diameter_mm, 3)


def assess_quality(image: Image, mask: Mask) -> dict[str, float]:
    """Acquisition quality, measured before the model is allowed to run.

    An unusable image must be rejected as unusable, not silently classified.
    """
    rows, cols = len(image), len(image[0])
    laplacian: list[float] = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            laplacian.append(
                abs(4 * image[r][c] - image[r - 1][c] - image[r + 1][c] - image[r][c - 1] - image[r][c + 1])
            )
    mean_lap = sum(laplacian) / len(laplacian) if laplacian else 0.0
    focus = sum((v - mean_lap) ** 2 for v in laplacian) / len(laplacian) if laplacian else 0.0
    flat = [v for row in image for v in row]
    clipped = sum(1 for v in flat if v <= 0.01 or v >= 0.99) / len(flat)
    area_fraction = sum(1 for row in mask for v in row if v) / float(rows * cols)
    return {
        "focus_score": round(focus * 1000.0, 4),
        "clipped_fraction": round(clipped, 4),
        "lesion_area_fraction": round(area_fraction, 4),
        "dynamic_range": round(_percentile(flat, 0.98) - _percentile(flat, 0.02), 4),
    }


def preprocess(dataset: Dataset, size: int = WORKING_SIZE) -> Preprocessed:
    """DICOM in, named features out. Each step is recorded for the audit trail."""
    steps: list[str] = []
    image = from_dataset(dataset)
    steps.append(f"window/level applied (C={dataset.get('WindowCenter')}, W={dataset.get('WindowWidth')})")
    image = median_filter(image)
    steps.append("3x3 median filter for hair and speckle suppression")
    original_size = len(image)
    image = resize(image, size)
    steps.append(f"bilinear resample {original_size}x{original_size} -> {size}x{size}")
    image = normalise_illumination(image)
    steps.append("illumination normalised to the 2nd-98th percentile")
    mask = segment(image)
    steps.append("Otsu threshold, largest 4-connected component retained")
    spacing = dataset.pixel_spacing_mm * (original_size / float(size))
    features, diameter_mm = extract_features(image, mask, spacing)
    steps.append(f"{len(FEATURE_NAMES)} named morphological features extracted")
    return Preprocessed(
        image=image,
        mask=mask,
        features=features,
        quality=assess_quality(image, mask),
        diameter_mm=diameter_mm,
        steps=steps,
    )


# ------------------------------------------------------------------ rendering
def encode_png(rows_rgb: list[list[tuple[int, int, int]]]) -> bytes:
    """Minimal PNG encoder so the dashboard can show the actual pixels.

    zlib and struct are standard library, so this keeps the no-dependency rule
    while still putting a real image in front of the reviewer.
    """
    height, width = len(rows_rgb), len(rows_rgb[0])
    raw = bytearray()
    for row in rows_rgb:
        raw.append(0)  # filter type 0 (None)
        for r, g, b in row:
            raw += bytes((r & 255, g & 255, b & 255))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def render(image: Image, mask: Mask | None = None, scale: int = 2) -> bytes:
    """Grayscale render with an optional segmentation outline drawn in amber."""
    rows, cols = len(image), len(image[0])
    outline: set[tuple[int, int]] = set()
    if mask is not None:
        for r in range(rows):
            for c in range(cols):
                if not mask[r][c]:
                    continue
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if not (0 <= nr < rows and 0 <= nc < cols) or not mask[nr][nc]:
                        outline.add((r, c))
                        break
    pixels: list[list[tuple[int, int, int]]] = []
    for r in range(rows):
        row: list[tuple[int, int, int]] = []
        for c in range(cols):
            if (r, c) in outline:
                row.append((213, 128, 59))
            else:
                value = int(round(min(1.0, max(0.0, image[r][c])) * 255))
                row.append((value, value, value))
        for _ in range(scale):
            pixels.append([p for p in row for _ in range(scale)])
    return encode_png(pixels)
