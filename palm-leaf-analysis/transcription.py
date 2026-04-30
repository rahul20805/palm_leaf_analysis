"""

The default mode creates a visual glyph-token transcript. It does not know the
language alphabet; instead it segments repeated glyph-like shapes, clusters
similar shapes, and writes tokens such as G001 G014. This is useful for an
assignment baseline when direct OCR is discouraged.

For real text output, pass a trained CRNN/CTC checkpoint with --mode ctc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import cv2
import numpy as np
try:  # Optional heavy dependency; allow module to be imported without torch
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


if TYPE_CHECKING:  # pragma: no cover - help static analyzers resolve the import
    from indic_transliteration import sanscript  # type: ignore
    # Optional heavy dependency; help static analyzers without importing at runtime
    try:  # pragma: no cover - type checker only
        import torch  # type: ignore
    except Exception:
        pass

try:
    import importlib

    sanscript = importlib.import_module("indic_transliteration.sanscript")
except Exception:  # pragma: no cover - transliteration is optional
    sanscript = None


SCRIPT_SCHEMES = {
    "bengali": "BENGALI",
    "devanagari": "DEVANAGARI",
    "gujarati": "GUJARATI",
    "gurmukhi": "GURMUKHI",
    "kannada": "KANNADA",
    "malayalam": "MALAYALAM",
    "oriya": "ORIYA",
    "tamil": "TAMIL",
    "telugu": "TELUGU",
}


@dataclass
class GlyphRecord:
    line_index: int
    bbox: list[int]
    feature: np.ndarray
    image: np.ndarray
    token: str = ""


def _load_segments(metadata_path: Path) -> list[dict[str, Any]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing segmentation metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return list(metadata.get("segments", []))


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read line crop: {path}")
    return image


def _binarize_line(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


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


def _merge_close_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in runs:
        if not merged or start - merged[-1][1] > max_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _glyph_boxes(binary: np.ndarray) -> list[list[int]]:
    height, width = binary.shape[:2]
    col_profile = np.count_nonzero(binary, axis=0)
    active = col_profile > max(1, int(height * 0.025))
    runs = _merge_close_runs(_runs_from_mask(active), max(2, height // 10))

    boxes: list[list[int]] = []
    for x1, x2 in runs:
        band = binary[:, x1 : x2 + 1]
        rows = np.where(np.count_nonzero(band, axis=1) > 0)[0]
        if rows.size == 0:
            continue
        y1 = max(0, int(rows[0]) - 2)
        y2 = min(height - 1, int(rows[-1]) + 2)
        x1 = max(0, x1 - 2)
        x2 = min(width - 1, x2 + 2)
        area = int(np.count_nonzero(binary[y1 : y2 + 1, x1 : x2 + 1]))
        if area < max(6, height * width * 0.0004):
            continue
        boxes.append([int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)])
    return boxes


def _glyph_feature(glyph_image: np.ndarray) -> np.ndarray:
    if glyph_image.size == 0:
        return np.zeros(256, dtype=np.float32)
    resized = cv2.resize(glyph_image, (16, 16), interpolation=cv2.INTER_AREA)
    vector = (resized.astype(np.float32) / 255.0).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def _collect_glyphs(segments: list[dict[str, Any]]) -> tuple[list[GlyphRecord], dict[int, list[GlyphRecord]]]:
    all_glyphs: list[GlyphRecord] = []
    by_line: dict[int, list[GlyphRecord]] = {}

    for segment in segments:
        line_index = int(segment["index"])
        gray = _read_gray(Path(segment["crop_path"]))
        binary = _binarize_line(gray)
        line_glyphs: list[GlyphRecord] = []

        for box in _glyph_boxes(binary):
            x, y, w, h = box
            glyph_image = binary[y : y + h, x : x + w]
            glyph = GlyphRecord(
                line_index=line_index,
                bbox=box,
                feature=_glyph_feature(glyph_image),
                image=glyph_image,
            )
            all_glyphs.append(glyph)
            line_glyphs.append(glyph)

        by_line[line_index] = line_glyphs
    return all_glyphs, by_line


def _assign_visual_tokens(glyphs: list[GlyphRecord], similarity_threshold: float = 0.78) -> None:
    centroids: list[np.ndarray] = []
    counts: list[int] = []

    for glyph in glyphs:
        best_index = -1
        best_score = -1.0
        for idx, centroid in enumerate(centroids):
            score = float(np.dot(glyph.feature, centroid))
            if score > best_score:
                best_index = idx
                best_score = score

        if best_index >= 0 and best_score >= similarity_threshold:
            counts[best_index] += 1
            centroids[best_index] = (
                centroids[best_index] * (counts[best_index] - 1) + glyph.feature
            ) / counts[best_index]
            norm = float(np.linalg.norm(centroids[best_index]))
            if norm > 0:
                centroids[best_index] /= norm
            glyph.token = f"G{best_index + 1:03d}"
        else:
            centroids.append(glyph.feature.copy())
            counts.append(1)
            glyph.token = f"G{len(centroids):03d}"


def _save_glyph_atlas(glyphs: list[GlyphRecord], output_path: Path) -> None:
    tile_w, tile_h = 96, 72
    cols = 6
    unique: dict[str, GlyphRecord] = {}
    for glyph in glyphs:
        unique.setdefault(glyph.token, glyph)

    rows = max(1, (len(unique) + cols - 1) // cols)
    atlas = np.full((rows * tile_h, cols * tile_w, 3), 255, dtype=np.uint8)

    if not unique:
        cv2.putText(atlas, "No glyphs found", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    else:
        for idx, (token, glyph) in enumerate(sorted(unique.items())):
            row, col = divmod(idx, cols)
            x0, y0 = col * tile_w, row * tile_h
            glyph_img = cv2.resize(glyph.image, (44, 36), interpolation=cv2.INTER_AREA)
            glyph_rgb = cv2.cvtColor(255 - glyph_img, cv2.COLOR_GRAY2BGR)
            atlas[y0 + 8 : y0 + 44, x0 + 26 : x0 + 70] = glyph_rgb
            cv2.putText(atlas, token, (x0 + 18, y0 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
            cv2.rectangle(atlas, (x0, y0), (x0 + tile_w - 1, y0 + tile_h - 1), (220, 220, 220), 1)

    cv2.imwrite(str(output_path), atlas)


def _transliterate(text: str, source_scheme: str | None, target_scheme: str = "iast") -> str | None:
    if not text or not source_scheme or sanscript is None:
        return None
    source_key = SCRIPT_SCHEMES.get(source_scheme.lower(), source_scheme.upper())
    target_key = target_scheme.upper()
    source = getattr(sanscript, source_key, None)
    target = getattr(sanscript, target_key, None)
    if source is None or target is None:
        return None
    return sanscript.transliterate(text, source, target)


def _read_vocab(vocab_path: Path | None, checkpoint: dict[str, Any] | None = None) -> list[str]:
    if checkpoint and isinstance(checkpoint.get("vocab"), list):
        return [str(item) for item in checkpoint["vocab"]]
    if vocab_path is None:
        raise ValueError("CTC mode requires --vocab unless the checkpoint stores a vocab list.")
    return [line.rstrip("\n") for line in vocab_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ctc_decode(indices: list[int], vocab: list[str]) -> str:
    chars: list[str] = []
    previous = -1
    for index in indices:
        if index != previous and index != 0:
            vocab_index = index - 1
            if 0 <= vocab_index < len(vocab):
                chars.append(vocab[vocab_index])
        previous = index
    return "".join(chars)


def _ctc_transcribe_line(crop_path: Path, checkpoint_path: Path, vocab_path: Path | None) -> tuple[str, str | None]:
    # torch is an optional heavy dependency imported at module level; if it's
    # unavailable, signal the user rather than attempting a local import which
    # some linters may flag as unresolved.
    if torch is None:
        return "", "PyTorch is required for CTC mode. Install torch and rerun."
    nn = torch.nn

    class TinyCRNN(nn.Module):
        def __init__(self, class_count: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 1), (2, 1)),
            )
            self.sequence = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
            self.classifier = nn.Linear(256, class_count)

        def forward(self, x: Any) -> Any:
            feats = self.features(x)
            feats = feats.mean(dim=2).permute(0, 2, 1)
            seq, _ = self.sequence(feats)
            return self.classifier(seq)

    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        vocab = _read_vocab(vocab_path, checkpoint if isinstance(checkpoint, dict) else None)
        model = TinyCRNN(class_count=len(vocab) + 1)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state)
        model.eval()

        gray = _read_gray(crop_path)
        height = 48
        width = max(16, int(gray.shape[1] * (height / max(1, gray.shape[0]))))
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            indices = logits.softmax(-1).argmax(-1).squeeze(0).tolist()
        return _ctc_decode(indices, vocab), None
    except Exception as exc:
        return "", str(exc)


def transcribe_segments(
    metadata_path: str | Path = "output/segments_meta.json",
    output_dir: str | Path = "output",
    mode: str = "glyph",
    source_scheme: str | None = None,
    target_scheme: str = "iast",
    ctc_checkpoint: str | Path | None = None,
    vocab_path: str | Path | None = None,
) -> dict:
    """Create a per-line transcript using the selected experimental method."""

    metadata_path = Path(metadata_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segments = _load_segments(metadata_path)
    mode = mode.lower()

    if mode not in {"glyph", "ctc"}:
        raise ValueError("mode must be either 'glyph' or 'ctc'")

    artifacts: dict[str, str] = {}
    results: list[dict[str, Any]] = []

    if mode == "glyph":
        glyphs, by_line = _collect_glyphs(segments)
        _assign_visual_tokens(glyphs)
        atlas_path = output_dir / "glyph_atlas.jpg"
        _save_glyph_atlas(glyphs, atlas_path)
        artifacts["glyph_atlas"] = str(atlas_path)

        for segment in segments:
            line_index = int(segment["index"])
            line_glyphs = by_line.get(line_index, [])
            text = " ".join(glyph.token for glyph in line_glyphs)
            results.append(
                {
                    "index": line_index,
                    "bbox": segment["bbox"],
                    "crop_path": segment["crop_path"],
                    "text": text,
                    "transliteration": None,
                    "engine": "visual-glyph-clustering",
                    "glyphs": [{"token": glyph.token, "bbox": glyph.bbox} for glyph in line_glyphs],
                    "error": None,
                }
            )
    else:
        if ctc_checkpoint is None:
            raise ValueError("CTC mode requires --ctc-checkpoint")
        checkpoint_path = Path(ctc_checkpoint)
        vocab_file = Path(vocab_path) if vocab_path else None
        for segment in segments:
            text, error = _ctc_transcribe_line(Path(segment["crop_path"]), checkpoint_path, vocab_file)
            results.append(
                {
                    "index": segment["index"],
                    "bbox": segment["bbox"],
                    "crop_path": segment["crop_path"],
                    "text": text,
                    "transliteration": _transliterate(text, source_scheme, target_scheme),
                    "engine": "tiny-crnn-ctc",
                    "glyphs": [],
                    "error": error,
                }
            )

    transcript = "\n".join(item["text"] for item in results)
    text_path = output_dir / "text.txt"
    json_path = output_dir / "transcription.json"
    text_path.write_text(transcript + ("\n" if transcript else ""), encoding="utf-8")
    artifacts["text"] = str(text_path)

    payload = {
        "line_count": len(results),
        "mode": mode,
        "note": (
            "Glyph mode is a no-direct-OCR visual-token baseline, not a true language transcript."
            if mode == "glyph"
            else "CTC mode expects a trained checkpoint for the selected script."
        ),
        "results": results,
        "artifacts": artifacts,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experimental no-direct-OCR palm-leaf transcription.")
    parser.add_argument("--metadata", default="output/segments_meta.json")
    parser.add_argument("--output", default="output")
    parser.add_argument("--mode", choices=["glyph", "ctc"], default="glyph")
    parser.add_argument("--ctc-checkpoint", default=None)
    parser.add_argument("--vocab", default=None)
    parser.add_argument("--script-source", default=None)
    args = parser.parse_args()

    result = transcribe_segments(
        args.metadata,
        args.output,
        mode=args.mode,
        source_scheme=args.script_source,
        ctc_checkpoint=args.ctc_checkpoint,
        vocab_path=args.vocab,
    )
    print(f"Transcribed {result['line_count']} line(s) with {result['mode']} mode.")
