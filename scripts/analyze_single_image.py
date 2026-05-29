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

from PIL import Image


WEIGHT_ALIASES = {
    "yolo_v1_default": ["elements-seg-v1-best.pt", "*v1*best*.pt"],
    "yolo_v2_aug_controlled": ["elements-seg-v2-aug-controlled-best.pt", "*v2*best*.pt"],
    "yolo_v3_no_erasing": ["elements-seg-v3-no-erasing-best.pt", "*v3*best*.pt", "*.pt"],
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
    hybrid_dir = session_dir / "hybrid_yolo_sam2"
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    raw = run_yolo(project_dir, session_dir, weights_dir, image_path, "yolo_v3_no_erasing", 0.25, yolo_device)
    raw_json = Path(raw["json"]) if raw.get("json") else None
    if raw_json is None:
        return {"name": "hybrid_yolo_sam2", "status": "missing yolo_v3", "json": None, "visual": None}

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


def write_reports(
    session_dir: Path,
    image_path: Path,
    results: list[dict[str, Any]],
    run_vlm: bool,
    api_key: str,
    vlm_model: str,
) -> dict[str, str]:
    report_dir = session_dir / "vlm_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "instruction": "Use total_detections and class_counts as authoritative counts. Do not recount samples.",
        "results": {item["name"]: item.get("counts", {}) for item in results},
    }
    (session_dir / "machine_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    reports: dict[str, str] = {}
    if not run_vlm:
        return reports

    prompts = {
        "image_only": "Bu cephe gorselini mimari cephe lejant raporu olarak yorumla. Sayilari tahminse belirt.",
        "model_assisted": (
            "Bu gorsel ve asagidaki model ciktilarina gore teknik mimari cephe lejant raporu yaz. "
            "YOLO versiyonlarini, SAM2/hibrit sonucu ve farklari kisa karsilastir. "
            "Sayi olarak sadece total_detections ve class_counts alanlarini kullan.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        ),
    }
    for name, prompt in prompts.items():
        text = call_vlm(image_path, prompt, api_key=api_key, model=vlm_model)
        path = report_dir / f"{name}.md"
        path.write_text(text, encoding="utf-8")
        reports[name] = str(path)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one uploaded image with YOLO/SAM2/VLM.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument("--reports-dir", required=True, type=Path)
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--run-v1", action="store_true")
    parser.add_argument("--run-v2", action="store_true")
    parser.add_argument("--run-v3", action="store_true")
    parser.add_argument("--run-hybrid", action="store_true")
    parser.add_argument("--sam2-dir", type=Path, default=None)
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-device", default="cuda")
    parser.add_argument("--run-vlm", action="store_true")
    parser.add_argument("--vlm-model", default="gpt-4o")
    args = parser.parse_args()

    session_name = args.session_name or f"single_image_{now_stamp()}_{uuid.uuid4().hex[:8]}"
    session_dir = args.reports_dir / "single_image_analysis" / session_name
    input_image = normalize_input_image(args.image, session_dir / "input")

    results: list[dict[str, Any]] = []
    if args.run_v1:
        results.append(run_yolo(args.project_dir, session_dir, args.weights_dir, input_image, "yolo_v1_default", 0.25, args.yolo_device))
    if args.run_v2:
        results.append(run_yolo(args.project_dir, session_dir, args.weights_dir, input_image, "yolo_v2_aug_controlled", 0.25, args.yolo_device))
    if args.run_v3:
        results.append(run_yolo(args.project_dir, session_dir, args.weights_dir, input_image, "yolo_v3_no_erasing", 0.25, args.yolo_device))
    if args.run_hybrid:
        results.append(
            run_hybrid(
                args.project_dir,
                session_dir,
                args.weights_dir,
                input_image,
                args.yolo_device,
                args.sam2_dir,
                args.sam2_checkpoint,
                args.sam2_model_cfg,
                args.sam2_device,
            )
        )

    reports = write_reports(
        session_dir=session_dir,
        image_path=input_image,
        results=results,
        run_vlm=args.run_vlm,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        vlm_model=args.vlm_model,
    )

    summary = {
        "session_dir": str(session_dir),
        "input_image": str(input_image),
        "results": results,
        "vlm_reports": reports,
    }
    summary_path = session_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== ANALYSIS SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
