"""Stage 1: detect text lines in a palm-leaf manuscript image.


"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    import cv2  # type: ignore[import]
except Exception as exc:  # pragma: no cover - helpful error if opencv missing
    raise ImportError(
        "opencv-python is required to run segmentation.py. Install it with: python -m pip install opencv-python"
    ) from exc
try:
    import numpy as np  # type: ignore[import]
except Exception as exc:  # pragma: no cover - helpful error if numpy missing
    raise ImportError(
        "numpy is required to run segmentation.py. Install it with: python -m pip install numpy"
    ) from exc


@dataclass
class LineSegment:
    index: int
    bbox: list[int]
    polygon: list[list[int]]
    crop_path: str


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _binarize_for_text(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        13,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32)
    window = min(window, max(1, len(values)))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _runs_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _merge_close_runs(runs: Iterable[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in runs:
        if not merged or start - merged[-1][1] > max_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _detect_line_boxes(
    image: np.ndarray,
    binary: np.ndarray,
    min_line_height: int = 12,
    gap_merge: int = 8,
    padding: int = 8,
    projection_threshold: float = 0.18,
) -> list[list[int]]:
    height, width = binary.shape[:2]
    projection = np.count_nonzero(binary, axis=1)
    smoothed = _smooth(projection, window=max(5, height // 120))
    threshold = max(3.0, smoothed.max() * projection_threshold)
    runs = _runs_from_mask(smoothed > threshold)
    runs = _merge_close_runs(runs, gap_merge)

    boxes: list[list[int]] = []
    for y1, y2 in runs:
        if y2 - y1 + 1 < min_line_height:
            continue

        y1 = max(0, y1 - padding)
        y2 = min(height - 1, y2 + padding)
        band = binary[y1 : y2 + 1, :]
        col_projection = np.count_nonzero(band, axis=0)
        active_cols = np.where(col_projection > max(2, band.shape[0] * 0.03))[0]
        if active_cols.size:
            x1 = max(0, int(active_cols[0]) - padding)
            x2 = min(width - 1, int(active_cols[-1]) + padding)
        else:
            x1, x2 = 0, width - 1

        boxes.append([x1, y1, x2 - x1 + 1, y2 - y1 + 1])

    if not boxes:
        boxes.append([0, 0, width, height])
    return boxes


def _save_projection_debug(binary: np.ndarray, output_path: Path) -> None:
    projection = np.count_nonzero(binary, axis=1).astype(np.float32)
    height = len(projection)
    width = 420
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    max_value = float(projection.max()) or 1.0
    for row, value in enumerate(projection):
        bar_width = int((value / max_value) * (width - 20))
        cv2.line(canvas, (10, row), (10 + bar_width, row), (30, 90, 220), 1)

    cv2.imwrite(str(output_path), canvas)


def _line_color(index: int) -> tuple[int, int, int]:
    hue = int((index * 41) % 180)
    hsv = np.uint8([[[hue, 190, 245]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _line_polygon(binary: np.ndarray, bbox: list[int]) -> list[list[int]]:
    x, y, w, h = bbox
    band = binary[y : y + h, x : x + w]
    if band.size == 0 or np.count_nonzero(band) == 0:
        return [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(9, w // 24), max(3, h // 4)),
    )
    connected = cv2.dilate(band, kernel, iterations=1)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]]

    points = np.vstack(contours)
    hull = cv2.convexHull(points)
    epsilon = max(2.0, 0.01 * cv2.arcLength(hull, True))
    polygon = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2)
    polygon[:, 0] += x
    polygon[:, 1] += y
    return [[int(px), int(py)] for px, py in polygon]


def segment_lines(
    image_path: str | Path,
    output_dir: str | Path = "output",
    min_line_height: int = 12,
    gap_merge: int = 8,
    padding: int = 8,
    projection_threshold: float = 0.18,
) -> dict:
    """Detect, crop, annotate, and serialize line segments."""

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    crop_dir = output_dir / "line_crops"
    _ensure_dir(output_dir)
    _ensure_dir(crop_dir)

    image = _read_image(image_path)
    binary = _binarize_for_text(image)
    boxes = _detect_line_boxes(
        image,
        binary,
        min_line_height=min_line_height,
        gap_merge=gap_merge,
        padding=padding,
        projection_threshold=projection_threshold,
    )

    annotated = image.copy()
    segments: list[LineSegment] = []
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        bbox = [int(x), int(y), int(w), int(h)]
        polygon = _line_polygon(binary, bbox)
        crop = image[y : y + h, x : x + w]
        crop_path = crop_dir / f"line_{idx:02d}.jpg"
        cv2.imwrite(str(crop_path), crop)

        color = _line_color(idx)
        points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [points], True, color, 2, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"Box {idx}",
            (x + 4, max(18, y + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        segments.append(LineSegment(idx, bbox, polygon, str(crop_path)))

    segmented_output = output_dir / "segmented_output.jpg"
    debug_projection = output_dir / "debug_projection.jpg"
    metadata_path = output_dir / "segments_meta.json"

    cv2.imwrite(str(segmented_output), annotated)
    _save_projection_debug(binary, debug_projection)

    metadata = {
        "source_image": str(image_path),
        "line_count": len(segments),
        "segments": [asdict(segment) for segment in segments],
        "artifacts": {
            "segmented_output": str(segmented_output),
            "debug_projection": str(debug_projection),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect and crop text lines from a palm-leaf image.")
    parser.add_argument("image", nargs="?", default="input/palm_leaf.jpg")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    result = segment_lines(args.image, args.output)
    print(f"Detected {result['line_count']} line(s).")
