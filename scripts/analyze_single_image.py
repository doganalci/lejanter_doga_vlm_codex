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


WEIGHTS = {
    "yolo_v1_default": ["elements-seg-v1-best.pt", "*v1*best*.pt"],
    "yolo_v2_aug_controlled": ["elements-seg-v2-aug-controlled-best.pt", "*v2*best*.pt"],
    "yolo_v3_no_erasing": ["elements-seg-v3-no-erasing-best.pt", "*v3*best*.pt", "*.pt"],
}

NAMES = {
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


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


def sh(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    done = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    out = "\n".join(part for part in (done.stdout, done.stderr) if part)
    if out:
        print(out, flush=True)
    if done.returncode:
        raise RuntimeError(f"Command failed with code {done.returncode}: {' '.join(cmd)}\n{out}")


def normalize_image(src: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "input.jpg"
    Image.open(src).convert("RGB").save(dst, quality=94)
    return dst


def label_image(src: Path, dst: Path, label: str) -> Path:
    image = Image.open(src).convert("RGB")
    w, h = image.size
    banner = max(54, int(h * 0.075))
    out = Image.new("RGB", (w, h + banner), (18, 18, 18))
    out.paste(image, (0, banner))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(22, int(banner * 0.42)))
    except OSError:
        font = ImageFont.load_default()
    draw.text((18, max(10, banner // 4)), label, fill=(255, 255, 255), font=font)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, quality=94)
    return dst


def first_visual(folder: Path) -> Path | None:
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        found = sorted(folder.glob(pattern))
        if found:
            return found[0]
    return None


def find_weight(weights_dir: Path, alias: str) -> Path | None:
    for pattern in WEIGHTS[alias]:
        found = sorted(weights_dir.rglob(pattern))
        if found:
            return found[0]
    return None


def records(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    return data if isinstance(data, list) else []


def counts(path: Path | None) -> dict[str, Any]:
    recs = records(path)
    dets = recs[0].get("detections", []) if recs else []
    labels = [item.get("label", "unknown") for item in dets]
    return {"total_detections": len(labels), "class_counts": dict(Counter(labels))}


def run_yolo(project: Path, session: Path, weights_dir: Path, image: Path, alias: str, device: str) -> dict[str, Any]:
    weight = find_weight(weights_dir, alias)
    model_dir = session / alias
    model_dir.mkdir(parents=True, exist_ok=True)
    if not weight:
        return {"name": alias, "status": "missing weight", "json": None, "visual": None}
    out_json = model_dir / "detections.json"
    runs = model_dir / "runs"
    cmd = [
        sys.executable, "scripts/infer_yolo.py",
        "--weights", str(weight), "--source", str(image), "--task", "segment",
        "--conf", "0.25", "--out", str(out_json), "--name", alias,
        "--project", str(runs), "--save-visuals",
    ]
    if device:
        cmd += ["--device", device]
    sh(cmd, project)
    visual = first_visual(runs / alias)
    if visual:
        copied = model_dir / visual.name
        shutil.copy2(visual, copied)
        visual = copied
    return {
        "name": alias, "status": "ok", "weight": str(weight),
        "json": str(out_json), "visual": str(visual) if visual else None,
        "counts": counts(out_json),
    }


def refine_sam(project: Path, sam2_dir: Path, detections: Path, checkpoint: Path, cfg: str, out_json: Path, visual_dir: Path, device: str) -> Path | None:
    sh([
        sys.executable, str(project / "scripts/refine_with_sam2.py"),
        "--detections", str(detections), "--checkpoint", str(checkpoint),
        "--model-cfg", cfg, "--out", str(out_json), "--visual-dir", str(visual_dir),
        "--device", device,
    ], sam2_dir)
    return first_visual(visual_dir)


def run_sam2(project: Path, session: Path, weights_dir: Path, image: Path, args: argparse.Namespace) -> dict[str, Any]:
    raw = run_yolo(project, session, weights_dir, image, "yolo_v3_no_erasing", args.yolo_device)
    raw_json = Path(raw["json"]) if raw.get("json") else None
    if not raw_json:
        return {"name": "sam2_from_yolo_v3", "status": "missing yolo_v3", "json": None, "visual": None}
    if not args.sam2_dir or not args.sam2_dir.exists() or not args.sam2_checkpoint or not args.sam2_checkpoint.exists():
        return {"name": "sam2_from_yolo_v3", "status": "SAM2 not installed", "json": str(raw_json), "visual": raw.get("visual"), "counts": counts(raw_json)}
    out_dir = session / "sam2_from_yolo_v3"
    out_json = out_dir / "sam2_from_yolo_v3.json"
    visual = refine_sam(project, args.sam2_dir, raw_json, args.sam2_checkpoint, args.sam2_model_cfg, out_json, out_dir / "visuals", args.sam2_device)
    return {"name": "sam2_from_yolo_v3", "status": "ok", "raw_json": str(raw_json), "json": str(out_json), "visual": str(visual) if visual else None, "counts": counts(out_json)}


def run_hybrid(project: Path, session: Path, weights_dir: Path, image: Path, args: argparse.Namespace) -> dict[str, Any]:
    raw = run_yolo(project, session, weights_dir, image, "yolo_v3_no_erasing", args.yolo_device)
    raw_json = Path(raw["json"]) if raw.get("json") else None
    if not raw_json:
        return {"name": "hybrid_yolo_sam2", "status": "missing yolo_v3", "json": None, "visual": None}
    out_dir = session / "hybrid_yolo_sam2"
    filtered = out_dir / "yolo_filtered_conf078.json"
    sh([
        sys.executable, "scripts/filter_detections.py", "--input", str(raw_json), "--out", str(filtered),
        "--default-conf", "0.78", "--class-threshold", "cam=0.82",
        "--class-threshold", "ahsap_dograma=0.78", "--class-threshold", "camur_harc=0.72",
        "--min-area-ratio", "0.00005", "--max-area-ratio", "0.35", "--nms-iou", "0.25",
    ], project)
    if not args.sam2_dir or not args.sam2_dir.exists() or not args.sam2_checkpoint or not args.sam2_checkpoint.exists():
        return {"name": "hybrid_yolo_sam2", "status": "filtered only; SAM2 not installed", "json": str(filtered), "visual": raw.get("visual"), "counts": counts(filtered)}
    out_json = out_dir / "hybrid_yolo_sam2.json"
    visual = refine_sam(project, args.sam2_dir, filtered, args.sam2_checkpoint, args.sam2_model_cfg, out_json, out_dir / "visuals", args.sam2_device)
    return {"name": "hybrid_yolo_sam2", "status": "ok", "raw_json": str(raw_json), "filtered_json": str(filtered), "json": str(out_json), "visual": str(visual) if visual else None, "counts": counts(out_json)}


def data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("utf-8")


def call_vlm(image: Path, prompt: str, api_key: str, model: str) -> str:
    if not api_key.strip():
        return "VLM raporu uretilmedi: API key girilmedi."
    import requests
    res = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model, "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "Turkce yanit veren mimari cephe lejant raporu asistanisin. Sayilari uydurma, verilen ozet sayilari esas al."},
                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_url(image)}}]},
            ],
        },
        timeout=180,
    )
    if res.status_code == 401:
        return "VLM raporu uretilmedi: OpenAI API key yetkisiz veya hatali."
    if res.status_code >= 400:
        return f"VLM raporu uretilmedi: HTTP {res.status_code}\n\n{res.text[:1200]}"
    return res.json()["choices"][0]["message"]["content"]


def final_assets(session: Path, image: Path, results: list[dict[str, Any]]) -> dict[str, str]:
    final = session / "final_report"
    image_dir, data_dir = final / "images", final / "data"
    image_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    images = {"input_image": str(label_image(image, image_dir / "00_input_original_image.jpg", NAMES["input_image"]))}
    for idx, item in enumerate(results, start=1):
        name, visual = item["name"], item.get("visual")
        if visual and Path(visual).exists():
            images[name] = str(label_image(Path(visual), image_dir / f"{idx:02d}_{slug(name)}_output.jpg", NAMES.get(name, name)))
        if item.get("json") and Path(item["json"]).exists():
            shutil.copy2(item["json"], data_dir / f"{idx:02d}_{slug(name)}.json")
    return images


def write_vlm_reports(session: Path, image: Path, results: list[dict[str, Any]], images: dict[str, str], run_vlm: bool, api_key: str, model: str) -> dict[str, str]:
    report_dir = session / "vlm_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    by_name = {item["name"]: item for item in results}
    machine = {"instruction": "Use total_detections and class_counts as authoritative counts.", "results": {item["name"]: item.get("counts", {}) for item in results}}
    (session / "machine_summary.json").write_text(json.dumps(machine, ensure_ascii=False, indent=2), encoding="utf-8")
    if not run_vlm:
        return {}
    specs = {
        "image_only": (image, "Verilen gorsel turu: sadece orijinal resim.", "Bu cephe gorselini mimari cephe lejant raporu olarak yorumla. Sayilari tahminse belirt."),
        "yolo_assisted": (Path(images.get("yolo_v3_no_erasing", image)), "Verilen gorsel turu: YOLO cikti gorseli + orijinal resim baglami + YOLO sayim ozeti.", json.dumps({k: by_name[k].get("counts", {}) for k in ("yolo_v1_default", "yolo_v2_aug_controlled", "yolo_v3_no_erasing") if k in by_name}, ensure_ascii=False, indent=2)),
        "sam2_assisted": (Path(images.get("sam2_from_yolo_v3", image)), "Verilen gorsel turu: SAM2 cikti gorseli + orijinal resim baglami + SAM2 sayim ozeti.", json.dumps({"sam2_from_yolo_v3": by_name.get("sam2_from_yolo_v3", {}).get("counts", {})}, ensure_ascii=False, indent=2)),
        "hybrid_yolo_sam2_assisted": (Path(images.get("hybrid_yolo_sam2", image)), "Verilen gorsel turu: hibrit filtreli YOLO + SAM2 cikti gorseli + orijinal resim baglami + hibrit sayim ozeti.", json.dumps({"hybrid_yolo_sam2": by_name.get("hybrid_yolo_sam2", {}).get("counts", {})}, ensure_ascii=False, indent=2)),
    }
    reports = {}
    for name, (vlm_image, context, payload) in specs.items():
        prompt = f"{context}\n\nMimari cephe lejant raporu yaz. Sayi olarak sadece verilen total_detections ve class_counts alanlarini kullan.\n\n{payload}"
        text = call_vlm(vlm_image, prompt, api_key, model)
        path = report_dir / f"{name}.md"
        path.write_text(f"# {NAMES.get(name, name)}\n\n**Verilen gorsel/deney baglami:** {context}\n\n**VLM'e verilen gorsel:** `{vlm_image}`\n\n{text}\n", encoding="utf-8")
        reports[name] = str(path)
    return reports


def write_final_report(session: Path, summary: dict[str, Any]) -> Path:
    final = session / "final_report"
    final.mkdir(parents=True, exist_ok=True)
    if (session / "machine_summary.json").exists():
        shutil.copy2(session / "machine_summary.json", final / "machine_summary.json")
    if summary.get("vlm_reports"):
        vlm_final = final / "vlm_reports"
        vlm_final.mkdir(exist_ok=True)
        for name, path in summary["vlm_reports"].items():
            if Path(path).exists():
                shutil.copy2(path, vlm_final / f"{slug(name)}.md")
    images = summary["final_report"]["images"]
    lines = ["# Cephe Lejant Tek Gorsel Analiz Raporu", "", f"Session dir: `{summary['session_dir']}`", f"Final report folder: `{final}`", "", "## 1. Girdi Gorseli", "", f"![input]({images.get('input_image', summary['input_image'])})", "", "## 2. Model Ciktilari", ""]
    for item in summary["results"]:
        name = item["name"]
        lines += [f"### {NAMES.get(name, name)}", "", f"Dosya/deney adi: `{name}`", f"Status: `{item.get('status')}`", "", "Counts:", "", "```json", json.dumps(item.get("counts", {}), ensure_ascii=False, indent=2), "```", ""]
        if images.get(name):
            lines += [f"![{name}]({images[name]})", ""]
    if summary.get("vlm_reports"):
        lines += ["## 3. VLM Raporlari", ""]
        for name, path in summary["vlm_reports"].items():
            lines += [f"### {NAMES.get(name, name)}", ""]
            if Path(path).exists():
                lines += [Path(path).read_text(encoding="utf-8"), ""]
    report = final / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(report, session / "analysis_report.md")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze one uploaded image with YOLO/SAM2/VLM.")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--project-dir", type=Path, default=Path.cwd())
    p.add_argument("--weights-dir", required=True, type=Path)
    p.add_argument("--reports-dir", required=True, type=Path)
    p.add_argument("--session-name", default=None)
    p.add_argument("--yolo-device", default="0")
    for flag in ("v1", "v2", "v3", "sam2", "hybrid", "vlm"):
        p.add_argument(f"--run-{flag}", action="store_true")
    p.add_argument("--sam2-dir", type=Path, default=None)
    p.add_argument("--sam2-checkpoint", type=Path, default=None)
    p.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    p.add_argument("--sam2-device", default="cuda")
    p.add_argument("--vlm-model", default="gpt-4o")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project, weights = args.project_dir, args.weights_dir
    session = args.reports_dir / "single_image_analysis" / (args.session_name or f"single_image_{stamp()}_{uuid.uuid4().hex[:8]}")
    image = normalize_image(args.image, session / "input")
    results: list[dict[str, Any]] = []
    if args.run_v1:
        results.append(run_yolo(project, session, weights, image, "yolo_v1_default", args.yolo_device))
    if args.run_v2:
        results.append(run_yolo(project, session, weights, image, "yolo_v2_aug_controlled", args.yolo_device))
    if args.run_v3:
        results.append(run_yolo(project, session, weights, image, "yolo_v3_no_erasing", args.yolo_device))
    if args.run_sam2:
        results.append(run_sam2(project, session, weights, image, args))
    if args.run_hybrid:
        results.append(run_hybrid(project, session, weights, image, args))
    images = final_assets(session, image, results)
    reports = write_vlm_reports(session, image, results, images, args.run_vlm, os.environ.get("OPENAI_API_KEY", ""), args.vlm_model)
    summary = {"session_dir": str(session), "input_image": str(image), "results": results, "vlm_reports": reports, "final_report": {"folder": str(session / "final_report"), "images": images}}
    summary_path = session / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_final_report(session, summary)
    summary["analysis_report"] = str(report)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(summary_path, Path(summary["final_report"]["folder"]) / "summary.json")
    print("\n=== ANALYSIS SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
