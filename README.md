Palm Leaf Analysis
Python pipeline for the assignment shown in the screenshots:

Palm leaf line segmentation with color-coded line regions.
Experimental transcription/transliteration without direct OCR by default.
Candidate damage detection with a JSON report.
Project Structure
palm-leaf-analysis/
|-- main.py
|-- segmentation.py
|-- transcription.py
|-- damage_detection.py
|-- requirements.txt
|-- README.md
|-- input/
|   `-- palm_leaf.jpg
`-- output/
    |-- segmented_output.jpg
    |-- line_crops/
    |-- segments_meta.json
    |-- debug_projection.jpg
    |-- text.txt
    |-- transcription.json
    |-- glyph_atlas.jpg
    |-- damage_map.jpg
    `-- damage_report.json
Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Outputs
output/segmented_output.jpg: original image with detected line polygons annotated.
output/line_crops/line_01.jpg: cropped images for each detected line.
output/segments_meta.json: bounding boxes, line polygons, and crop paths.
output/debug_projection.jpg: horizontal projection profile used for line detection.
output/text.txt: per-line glyph tokens or CTC text.
output/transcription.json: per-line transcript, glyph boxes, engine, and errors.
output/glyph_atlas.jpg: representative visual token atlas for glyph mode.
output/damage_map.jpg: original image with candidate damage regions highlighted.
output/damage_report.json: damage region statistics and bounding boxes.
Notes
Line detection assumes the image is reasonably upright with mostly horizontal writing. If the manuscript is rotated or curved, deskew or crop it first for better results.

Damage detection is image-based and reports candidate regions. It is useful for triage, but final conservation judgments should be reviewed by a domain expert.
