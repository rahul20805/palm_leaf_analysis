"""Stage 3: detect visible damage or surface anomalies in the manuscript image."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import cv2  # type: ignore
except Exception as e:
    raise ImportError(
        "OpenCV (cv2) is required to run this script. Install it with: pip install opencv-python"
    ) from e
try:
    import numpy as np  # type: ignore
except Exception as e:
    raise ImportError(
        "NumPy is required to run this script. Install it with: pip install numpy"
    ) from e


@dataclass
class DamageRegion:
    index: int
    bbox: list[int]
    area_pixels: int
    area_percent: float
    kind: str


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _build_damage_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    background = cv2.medianBlur(blurred, 31)
    diff = cv2.absdiff(blurred, background)

    diff_cutoff = max(18, int(np.percentile(diff, 94)))
    dark_cutoff = int(np.percentile(gray, 7))
    light_cutoff = int(np.percentile(gray, 97))
    low_sat_cutoff = int(np.percentile(saturation, 45))

    anomaly_mask = (diff > diff_cutoff).astype(np.uint8) * 255
    dark_mask = (gray < dark_cutoff).astype(np.uint8) * 255
    light_mask = ((gray > light_cutoff) & (saturation < low_sat_cutoff)).astype(np.uint8) * 255

    mask = cv2.bitwise_or(anomaly_mask, dark_mask)
    mask = cv2.bitwise_or(mask, light_mask)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)


def _classify_region(gray_crop: np.ndarray, global_gray: np.ndarray) -> str:
    mean_value = float(np.mean(gray_crop))
    if mean_value < float(np.percentile(global_gray, 12)):
        return "dark_stain_or_loss"
    if mean_value > float(np.percentile(global_gray, 88)):
        return "abrasion_or_light_patch"
    return "surface_anomaly"


def detect_damage(
    image_path: str | Path,
    output_dir: str | Path = "output",
    min_area: int = 150,
) -> dict:
    """Detect candidate damage regions and save an annotated map/report."""

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = _read_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = _build_damage_mask(image)
    image_area = int(image.shape[0] * image.shape[1])

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[DamageRegion] = []
    annotated = image.copy()
    overlay = image.copy()
    overlay[mask > 0] = (0, 0, 255)
    annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)

    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area or area > image_area * 0.35:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        crop = gray[y : y + h, x : x + w]
        kind = _classify_region(crop, gray)
        region = DamageRegion(
            index=len(regions) + 1,
            bbox=[int(x), int(y), int(w), int(h)],
            area_pixels=area,
            area_percent=round((area / image_area) * 100.0, 4),
            kind=kind,
        )
        regions.append(region)
        cv2.rectangle(annotated, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            str(region.index),
            (x + 4, max(18, y + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    damage_map = output_dir / "damage_map.jpg"
    report_path = output_dir / "damage_report.json"
    cv2.imwrite(str(damage_map), annotated)

    total_area = sum(region.area_pixels for region in regions)
    report = {
        "source_image": str(image_path),
        "damage_region_count": len(regions),
        "damage_area_pixels": total_area,
        "damage_area_percent": round((total_area / image_area) * 100.0, 4),
        "regions": [asdict(region) for region in regions],
        "artifacts": {"damage_map": str(damage_map)},
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect visible damage in a palm-leaf image.")
    parser.add_argument("image", nargs="?", default="input/palm_leaf.jpg")
    parser.add_argument("--output", default="output")
    parser.add_argument("--min-area", type=int, default=150)
    args = parser.parse_args()

    result = detect_damage(args.image, args.output, min_area=args.min_area)
    print(f"Detected {result['damage_region_count']} damage region(s).")
