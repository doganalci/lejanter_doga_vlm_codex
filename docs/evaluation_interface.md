# Evaluation Interface

Bu arayuz mevcut egitim ve model uretme akisini degistirmeden kolay test, karsilastirma ve insan degerlendirmesi yapmak icin eklendi.

## Goal

```text
input image
  -> YOLO v1 / YOLO v2 / YOLO v3
  -> hybrid YOLO + filter + SAM2
  -> optional VLM reports
  -> visual comparison
  -> score/comment collection
  -> Drive + SQLite persistence
```

## Files

```text
scripts/evaluation_interface.py
notebooks/evaluation_interface_colab.ipynb
```

## Storage Layout

Colab notebook Drive'daki `reports/` klasorunu kullanir:

```text
reports/
  evaluation_sessions/
    ratings.sqlite
    session_YYYYMMDD_HHMMSS_xxxxxxxx/
      input/
        input.jpg
      yolo_v1_default/
      yolo_v2_aug_controlled/
      yolo_v3_no_erasing/
      hybrid_yolo_sam2/
      vlm_reports/
      session_summary.md
```

## SQLite Tables

```text
sessions
  session_id
  created_at
  input_image_path
  session_dir

artifacts
  session_id
  artifact_type
  name
  path
  metadata_json

ratings
  session_id
  target_name
  score
  comment
```

## Current Scope

The first version is a Colab-hosted Gradio prototype. This keeps GPU, SAM2, YOLO weights, Drive storage, and VLM API calls in the environment where the project already works.

Supported evaluations:

- YOLO v1 default model
- YOLO v2 controlled augmentation model
- YOLO v3 no-erasing model
- Hybrid YOLO v3 + stricter filtering + SAM2
- VLM image-only report
- VLM detection-assisted report

## Next Steps

- Add per-class threshold controls in the UI.
- Add side-by-side metric cards for class counts.
- Add export as a single HTML/PDF evaluation report.
- Add reviewer identity if multiple people score the same result.
- Later, move from Colab Gradio to FastAPI + React if a persistent web service is needed.
