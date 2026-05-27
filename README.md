# Lejanter Doga VLM Codex

Facade legend detection, facade segmentation, and report generation pipeline.

The intended flow is:

```text
Roboflow labeled data
  -> facade segmentation model
  -> facade element detection/segmentation model
  -> spatial join: which element belongs to which facade
  -> JSON/CSV outputs
  -> vLLM-compatible architectural facade report
```

## Why this shape

For this project, the primary model should be a detector/segmenter, not a VLM fine-tuned first. Bounding boxes and masks are measurable, easier to debug, and map directly to architectural legend counts. The VLM/LLM layer is best used after that to turn structured detections into a clean report.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Colab

Use [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) for the first GPU baseline. In Colab, choose:

```text
Runtime -> Change runtime type -> GPU
GPU preference: A100 > L4 > T4
```

The Colab notebook can also pull data from the shared Google Drive folder:

```text
dataset/  -> Roboflow YOLO zip
weights/  -> trained .pt model weights
test/     -> external test images for visual comparison
reports/  -> exported runs, metrics, configs, and benchmark visuals
```

## Technical Report

See [docs/technical_report.md](docs/technical_report.md) for the detailed method report, training plan, expected durations, metric definitions, and evaluation checklist.

## Data

Export Roboflow datasets in YOLO format:

1. Facade segmentation dataset: masks/polygons for visible facade regions.
2. Element dataset: windows, panels, doors, railings, legend elements, etc.

Copy the templates and edit the paths/classes:

```bash
cp configs/facade_dataset.template.yaml configs/facade_dataset.yaml
cp configs/elements_dataset.template.yaml configs/elements_dataset.yaml
```

Recommended first labeling rule: use one class, `facade_region`, and label each visible facade plane as a separate instance.

Current local dataset:

```text
data/roboflow/gtu-mimari-lejant
  train: 42 images
  valid: 5 images
  test: 5 images
  classes: 17
```

## Train

Facade segmentation:

```bash
python scripts/train_yolo.py \
  --data configs/facade_dataset.yaml \
  --model yolo11n-seg.pt \
  --task segment \
  --name facade-seg-v1 \
  --epochs 80 \
  --imgsz 1024
```

Element detection:

```bash
python scripts/train_yolo.py \
  --data configs/elements_dataset.yaml \
  --model yolo11s.pt \
  --task detect \
  --name elements-det-v1 \
  --epochs 100 \
  --imgsz 1024
```

Element segmentation, if the labels are polygon/mask based:

```bash
python scripts/train_yolo.py \
  --data configs/elements_dataset.yaml \
  --model yolo11s-seg.pt \
  --task segment \
  --name elements-seg-v1 \
  --epochs 100 \
  --imgsz 1024
```

## Inference

Run facade and element models on the same folder:

```bash
python scripts/infer_yolo.py \
  --weights outputs/runs/facade-seg-v1/weights/best.pt \
  --source data/raw/test_images \
  --task segment \
  --out outputs/reports/facades.json \
  --save-visuals

python scripts/infer_yolo.py \
  --weights outputs/runs/elements-det-v1/weights/best.pt \
  --source data/raw/test_images \
  --task detect \
  --out outputs/reports/elements.json \
  --save-visuals
```

Create a first report from detections:

```bash
python scripts/report_from_detections.py \
  --elements outputs/reports/elements.json \
  --facades outputs/reports/facades.json \
  --out outputs/reports/facade_report.md
```

Optional vLLM report, using any OpenAI-compatible vLLM server:

```bash
python scripts/report_with_vllm.py \
  --elements outputs/reports/elements.json \
  --facades outputs/reports/facades.json \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --out outputs/reports/vllm_report.md
```

## Next Decisions

- Confirm whether Roboflow labels are boxes or polygons.
- Finalize the element class list.
- Label `facade_region` masks if they do not exist yet.
- Train a small baseline, inspect 20 outputs, then decide whether to scale model size.
