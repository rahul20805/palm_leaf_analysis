"""Run the full palm-leaf manuscript analysis pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from damage_detection import detect_damage
from segmentation import segment_lines
from transcription import transcribe_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Palm-leaf manuscript segmentation, transcription, and damage analysis.")
    parser.add_argument("--input", default="input/palm_leaf.jpg", help="Path to the manuscript image.")
    parser.add_argument("--output", default="output", help="Directory for generated artifacts.")
    parser.add_argument(
        "--transcription-mode",
        choices=["glyph", "ctc"],
        default="glyph",
        help="glyph creates no-OCR visual tokens; ctc uses a trained CRNN/CTC checkpoint.",
    )
    parser.add_argument("--ctc-checkpoint", default=None, help="Path to a trained CRNN/CTC checkpoint.")
    parser.add_argument("--vocab", default=None, help="UTF-8 vocabulary file for CTC mode, one character per line.")
    parser.add_argument(
        "--script-source",
        default=None,
        help="Optional source script for transliterating CTC output, for example devanagari, tamil, kannada.",
    )
    parser.add_argument(
        "--skip-transcription",
        "--skip-ocr",
        action="store_true",
        help="Run segmentation and damage detection only.",
    )
    parser.add_argument("--min-line-height", type=int, default=12)
    parser.add_argument("--gap-merge", type=int, default=8)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--projection-threshold", type=float, default=0.18)
    parser.add_argument("--damage-min-area", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.input)
    output_dir = Path(args.output)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}. Put your image at input/palm_leaf.jpg "
            "or pass a custom path with --input."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Stage 1/3: segmenting manuscript lines...")
    segmentation = segment_lines(
        image_path,
        output_dir,
        min_line_height=args.min_line_height,
        gap_merge=args.gap_merge,
        padding=args.padding,
        projection_threshold=args.projection_threshold,
    )
    print(f"  detected {segmentation['line_count']} line(s)")
    

    if args.skip_transcription:
        print("Stage 2/3: transcription skipped")
    else:
        print("Stage 2/3: transcribing line crops...")
        transcription = transcribe_segments(
            output_dir / "segments_meta.json",
            output_dir,
            mode=args.transcription_mode,
            source_scheme=args.script_source,
            ctc_checkpoint=args.ctc_checkpoint,
            vocab_path=args.vocab,
        )
        errored = sum(1 for item in transcription["results"] if item.get("error"))
        print(
            f"  transcribed {transcription['line_count']} line(s) with "
            f"{transcription['mode']} mode; {errored} line(s) reported errors"
        )

    print("Stage 3/3: detecting damage regions...")
    damage = detect_damage(image_path, output_dir, min_area=args.damage_min_area)
    print(f"  detected {damage['damage_region_count']} damage region(s)")

    print(f"\nDone. Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
