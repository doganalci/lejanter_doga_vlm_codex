from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


WEIGHT_ALIASES = {
    "yolo_v1_default": ["elements-seg-v1-best.pt", "*v1*best*.pt"],
    "yolo_v2_aug_controlled": ["elements-seg-v2-aug-controlled-best.pt", "*v2*best*.pt"],
    "yolo_v3_no_erasing": ["elements-seg-v3-no-erasing-best.pt", "*v3*best*.pt", "*.pt"],
}

DISPLAY_NAMES = {
    "input_image": "Original Input Image",
    "yolo_v1_default": "YOLO v1 Default",
    "yolo_v2_aug_controlled": "YOLO v2 Aug Controlled",
    "yolo_v3_no_erasing": "YOLO v3 No Erasing",
    "sam2_from_yolo_v3": "SAM2 From YOLO v3",
    "hybrid_yolo_sam2": "Hybrid Filtered YOLO v3 + SAM2",
    "image_only": "VLM: Sadece Resim",
    "yolo_assisted": "VLM: Resim + YOLO Ciktilari",
    "sam2_assisted": "VLM: Resim + SAM2 Ciktisi",
    "hybrid_yolo_sam2_assisted": "VLM: Resim + Hibrit Cikti",
}


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(command: list[str], cwd: Path) -> str:
    print("\n$ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    if output:
        print(output, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(command)}\n{output}")
    return output


def find_weight(weights_dir: Path, alias: str) -> Path | None:
    for pattern in WEIGHT_ALIASES[alias]:
        matches = sorted(weights_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def normalize_input_image(source_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "input.jpg"
    Image.open(source_path).convert("RGB").save(target, quality=94)
    return target


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def label_image(source_path: Path, target_path: Path, label: str) -> Path:
    image = Image.open(source_path).convert("RGB")
    width, height = image.size
    banner_height = max(54, int(height * 0.075))
    output = Image.new("RGB", (width, height + banner_height), (18, 18, 18))
    output.paste(image, (0, banner_height))
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(22, int(banner_height * 0.42)))
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, width, banner_height], fill=(18, 18, 18))
    draw.text((18, max(10, banner_height // 4)), label, fill=(255, 255, 255), font=font)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(target_path, quality=94)
    return target_path


def first_visual(run_dir: Path) -> Path | None:
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        matches = sorted(run_dir.glob(suffix))
        if matches:
            return matches[0]
    return None


def load_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    return payload if isinstance(payload, list) else []


def detection_counts(path: Path | None) -> dict[str, Any]:
    records = load_records(path)
    detections = records[0].get("detections", []) if records else []
    labels = [item.get("label", "unknown") for item in detections]
    return {"total_detections": len(labels), "class_counts": dict(Counter(labels))}


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def call_vlm(image_path: Path, prompt: str, api_key: str, model: str) -> str:
    if not api_key.strip():
        return "VLM raporu uretilmedi: API key girilmedi."

    import requests

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Turkce yanit veren mimari cephe lejant raporu asistanisin. "
                        "Sayilari uydurma, verilen ozet sayilari esas al."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                    ],
                },
            ],
        },
        timeout=180,
    )
    if response.status_code == 401:
        return "VLM raporu uretilmedi: OpenAI API key yetkisiz veya hatali."
    if response.status_code >= 400:
        return f"VLM raporu uretilmedi: HTTP {response.status_code}\n\n{response.text[:1200]}"
    return response.json()["choices"][0]["message"]["content"]


def run_yolo(
    project_dir: Path,
    session_dir: Path,
    weights_dir: Path,
    image_path: Path,
    alias: str,
    conf: float,
    device: str,
) -> dict[str, Any]:
    weight = find_weight(weights_dir, alias)
    model_dir = session_dir / alias
    model_dir.mkdir(parents=True, exist_ok=True)
    if weight is None:
        return {"name": alias, "status": "missing weight", "json": None, "visual": None}

    out_json = model_dir / "detections.json"
    run_name = alias
    project = model_dir / "runs"
    command = [
        sys.executable,
        "scripts/infer_yolo.py",
        "--weights",
        str(weight),
        "--source",
        str(image_path),
        "--task",
        "segment",
        "--conf",
        str(conf),
        "--out",
        str(out_json),
        "--name",
        run_name,
        "--project",
        str(project),
        "--save-visuals",
    ]
    if device:
        command.extend(["--device", device])
    run_command(command, project_dir)

    visual = first_visual(project / run_name)
    if visual:
        copied = model_dir / visual.name
        shutil.copy2(visual, copied)
        visual = copied
    return {
        "name": alias,
        "status": "ok",
        "weight": str(weight),
        "json": str(out_json),
        "visual": str(visual) if visual else None,
        "counts": detection_counts(out_json),
    }


def run_hybrid(
    project_dir: Path,
    session_dir: Path,
    weights_dir: Path,
    image_path: Path,
    yolo_device: str,
    sam2_dir: Path | None,
    sam2_checkpoint: Path | None,
    sam2_model_cfg: str,
    sam2_device: str,
) -> dict[str, Any]:
    raw = run_yolo(project_dir, session_dir, weights_dir, image_path, "yolo_v3_no_erasing", 0.25, yolo_device)
    raw_json = Path(raw["json"]) if raw.get("json") else None
    if raw_json is None:
        return {"name": "hybrid_yolo_sam2", "status": "missing yolo_v3", "json": None, "visual": None}

    hybrid_dir = session_dir / "hybrid_yolo_sam2"
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    filtered_json = hybrid_dir / "yolo_filtered_conf078.json"
    run_command(
        [
            sys.executable,
            "scripts/filter_detections.py",
            "--input",
            str(raw_json),
            "--out",
            str(filtered_json),
            "--default-conf",
            "0.78",
            "--class-threshold",
            "cam=0.82",
            "--class-threshold",
            "ahsap_dograma=0.78",
            "--class-threshold",
            "camur_harc=0.72",
            "--min-area-ratio",
            "0.00005",
            "--max-area-ratio",
            "0.35",
            "--nms-iou",
            "0.25",
        ],
        project_dir,
    )

    if not sam2_dir or not sam2_dir.exists() or not sam2_checkpoint or not sam2_checkpoint.exists():
        return {
            "name": "hybrid_yolo_sam2",
            "status": "filtered only; SAM2 not installed",
            "raw_json": str(raw_json),
            "json": str(filtered_json),
            "visual": raw.get("visual"),
            "counts": detection_counts(filtered_json),
        }

    hybrid_json = hybrid_dir / "hybrid_yolo_sam2.json"
    visual_dir = hybrid_dir / "visuals"
    run_command(
        [
            sys.executable,
            str(project_dir / "scripts/refine_with_sam2.py"),
            "--detections",
            str(filtered_json),
            "--checkpoint",
            str(sam2_checkpoint),
            "--model-cfg",
            sam2_model_cfg,
            "--out",
            str(hybrid_json),
            "--visual-dir",
            str(visual_dir),
            "--device",
            sam2_device,
        ],
        sam2_dir,
    )
    visual = first_visual(visual_dir)
    return {
        "name": "hybrid_yolo_sam2",
        "status": "ok",
        "raw_json": str(raw_json),
        "filtered_json": str(filtered_json),
        "json": str(hybrid_json),
        "visual": str(visual) if visual else None,
        "counts": detection_counts(hybrid_json),
    }


def run_sam2_from_yolo(
    project_dir: Path,
    session_dir: Path,
    weights_dir: Path,
    image_path: Path,
    yolo_device: str,
    sam2_dir: Path | None,
    sam2_checkpoint: Path | None,
    sam2_model_cfg: str,
    sam2_device: str,
) -> dict[str, Any]:
    raw = run_yolo(project_dir, session_dir, weights_dir, image_path, "yolo_v3_no_erasing", 0.25, yolo_device)
    raw_json = Path(raw["json"]) if raw.get("json") else None
    if raw_json is None:
        return {"name": "sam2_from_yolo_v3", "status": "missing yolo_v3", "json": None, "visual": None}

    if not sam2_dir or not sam2_dir.exists() or not sam2_checkpoint or not sam2_checkpoint.exists():
        return {
            "name": "sam2_from_yolo_v3",
            "status": "SAM2 not installed; raw YOLO returned",
            "raw_json": str(raw_json),
            "json": str(raw_json),
            "visual": raw.get("visual"),
            "counts": detection_counts(raw_json),
        }

    sam_dir = session_dir / "sam2_from_yolo_v3"
    sam_dir.mkdir(parents=True, exist_ok=True)
    sam_json = sam_dir / "sam2_from_yolo_v3.json"
    visual_dir = sam_dir / "visuals"
    run_command(
        [
            sys.executable,
            str(project_dir / "scripts/refine_with_sam2.py"),
            "--detections",
            str(raw_json),
            "--checkpoint",
            str(sam2_checkpoint),
            "--model-cfg",
            sam2_model_cfg,
            "--out",
            str(sam_json),
            "--visual-dir",
            str(visual_dir),
            "--device",
            sam2_device,
        ],
        sam2_dir,
    )
    visual = first_visual(visual_dir)
    return {
        "name": "sam2_from_yolo_v3",
        "status": "ok",
        "raw_json": str(raw_json),
        "json": str(sam_json),
        "visual": str(visual) if visual else None,
        