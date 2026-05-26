# Roadmap

## Phase 1: Baseline

- Export Roboflow data in YOLO format.
- Train facade segmentation baseline.
- Train element detection or segmentation baseline.
- Produce visual overlays and JSON outputs.

## Phase 2: Geometry Join

- Assign each detected element to the facade mask containing its center point.
- Count elements per facade.
- Flag detections outside any facade as review candidates.

## Phase 3: Reporting

- Generate Markdown/CSV summaries.
- Add vLLM-compatible report generation.
- Standardize report language for architectural facade legend reviews.

## Phase 4: Improve

- Inspect errors by class.
- Add labels for missing facade cases.
- Increase model size only after data quality is stable.
- Consider VLM fine-tuning only if structured detection plus prompting is insufficient.
