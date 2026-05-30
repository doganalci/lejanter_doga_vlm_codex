from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from evaluation_interface import (
    EvaluationRunner,
    VLM_MODELS,
    call_vlm,
    detection_counts,
    find_weight,
    first_visual,
    insert_artifact,
    insert_rating,
    insert_session,
    make_report_payload,
    make_yolo_payload,
    normalize_input_image,
    one_record,
    save_vlm_report,
)

CLASS_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (57, 59, 121), (82, 84, 163),
    (107, 110, 207), (156, 158, 222), (99, 121, 57), (140, 162, 82),
    (181, 207, 107), (206, 219, 156), (140, 109, 49), (189, 158, 57),
    (231, 186, 82), (231, 203, 148), (132, 60, 57), (173, 73, 74),
]

TITLE_MAP = {
    "overall": "Genel yorum ve puan",
    "yolo_v1_default": "YOLO v1 sonucu",
    "yolo_v2_aug_controlled": "YOLO v2 augmentasyonlu sonuc",
    "yolo_v3_no_erasing": "YOLO v3 no-erasing sonuc",
    "hybrid_yolo_sam2": "Hibrit YOLO + SAM2 sonucu",
    "vlm_image_only": "LLM raporu: sadece resim",
    "vlm_yolo_assisted": "LLM raporu: resim + YOLO tespit ozeti",
    "vlm_sam2_assisted": "LLM raporu: resim + SAM2 rafine ozeti",
    "vlm_hybrid_yolo_sam2": "LLM raporu: resim + hibrit YOLO + SAM2 ozeti",
}


def title_for(target: str) -> str:
    return TITLE_MAP.get(target, target)


def class_color(label: str) -> tuple[int, int, int]:
    stable_index = sum((index + 1) * ord(char) for index, char in enumerate(label))
    return CLASS_COLORS[stable_index % len(CLASS_COLORS)]


def read_record(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data[0] if isinstance(data, list) and data else data
    except Exception:
        return None


def render_detection_visual(image_path: Path, record_path: Path, out_path: Path, style: str) -> Path | None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    record = read_record(record_path)
    detections = record.get("detections", []) if isinstance(record, dict) else []
    if not detections:
        return None
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for index, det in enumerate(detections, start=1):
        label = str(det.get("label", "unknown"))
        color = class_color(label)
        polygon = det.get("polygon") or []
        bbox = det.get("bbox_xyxy") or []
        if polygon and style in {"mask_only", "small_labels"}:
            points = [(float(x), float(y)) for x, y in polygon]
            draw.polygon(points, fill=(*color, 70), outline=(*color, 230))
        if len(bbox) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            draw.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=3)
            if style == "small_labels":
                text = f"{index}. {label} {det.get('confidence', '')}"
                width = int(draw.textlength(text, font=font)) + 8
                top = max(0, y1 - 20)
                draw.rectangle([x1, top, x1 + width, top + 20], fill=(*color, 215))
                draw.text((x1 + 4, top + 2), text, fill=(255, 255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(out_path, quality=92)
    return out_path


def render_legend_image(record_path: Path, out_path: Path, title: str) -> Path | None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    record = read_record(record_path)
    detections = record.get("detections", []) if isinstance(record, dict) else []
    if not detections:
        return None
    counts: dict[str, int] = {}
    for det in detections:
        label = str(det.get("label", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    try:
        title_font = ImageFont.truetype("Arial.ttf", 22)
        font = ImageFont.truetype("Arial.ttf", 18)
    except Exception:
        title_font = ImageFont.load_default()
        font = ImageFont.load_default()
    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    image = Image.new("RGB", (520, max(90, 58 + 30 * len(rows))), "white")
    draw = ImageDraw.Draw(image)
    draw.text((18, 16), f"Renk Cetveli: {title}", fill=(20, 20, 20), font=title_font)
    y = 56
    for label, count in rows:
        color = class_color(label)
        draw.rectangle([20, y + 5, 42, y + 27], fill=color)
        draw.text((54, y + 3), f"{label}: {count}", fill=(20, 20, 20), font=font)
        y += 30
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=92)
    return out_path


def make_visual_summary(gallery: list[tuple[str, str]], out_path: Path) -> Path | None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    items = []
    for image_name, label in gallery:
        path = Path(image_name)
        if not path.exists():
            continue
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((360, 240))
            items.append((image.copy(), label))
        except Exception:
            continue
    if not items:
        return None
    cols = 2 if len(items) <= 4 else 3
    pad, label_h, cell_w, cell_h = 18, 34, 396, 310
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for index, (image, label) in enumerate(items):
        col, row = index % cols, index // cols
        x0, y0 = col * cell_w, row * cell_h
        draw.rectangle([x0 + 6, y0 + 6, x0 + cell_w - 6, y0 + cell_h - 6], outline=(210, 210, 210), width=2)
        draw.text((x0 + pad, y0 + pad), label[:42], fill=(20, 20, 20), font=font)
        sheet.paste(image, (x0 + pad + (360 - image.width) // 2, y0 + pad + label_h + (240 - image.height) // 2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
    return out_path


def write_combined_report(session_dir: Path, session_id: str, visual_summary: Path | None, details: list[dict[str, str]], feedback_map: dict[str, dict[str, Any]] | None = None) -> Path:
    parts = ["# Birlesik Cephe Lejant Raporu", "", f"Session: `{session_id}`", ""]
    if visual_summary:
        parts += ["## Toplu Gorsel Sonuc", "", f"![Toplu gorsel sonuc]({visual_summary.relative_to(session_dir).as_posix()})", "", "----", ""]
    parts += ["## Sonuc Detaylari", ""]
    for detail in details:
        target = detail.get("target", "")
        title = detail.get("title", target)
        parts += [f"### {title}", "", f"Target: `{target}`", ""]
        for key, label in (("image", "Gorsel"), ("legend", "Renk cetveli")):
            value = detail.get(key, "")
            if value:
                try:
                    rel = Path(value).relative_to(session_dir).as_posix()
                except ValueError:
                    rel = value
                parts += [f"**{label}**", "", f"![{title} {label}]({rel})", ""]
        if detail.get("content"):
            parts += [detail["content"], ""]
        feedback = (feedback_map or {}).get(target)
        if feedback:
            parts += ["**Puan ve yorum**", "", f"- Puan: `{feedback.get('score') if feedback.get('score') is not None else 'no feedback'}`", f"- Yorum: {feedback.get('comment') or 'no feedback'}", ""]
        parts += ["----", ""]
    out_path = session_dir / "combined_report.md"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def get_openai_api_key() -> tuple[str, str]:
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key, "OPENAI_API_KEY ortam degiskeninden alindi."
    try:
        from google.colab import userdata  # type: ignore
        secret_key = (userdata.get("OPENAI_API_KEY") or "").strip()
        if secret_key:
            os.environ["OPENAI_API_KEY"] = secret_key
            return secret_key, "Colab Secrets icindeki OPENAI_API_KEY alindi."
    except Exception:
        pass
    return "", "OPENAI_API_KEY bulunamadi. Colab Secrets'ta OPENAI_API_KEY ekleyip Notebook access'i ac."


def live_run_command(command: list[str], cwd: Path, log_callback: Any | None = None) -> str:
    if log_callback:
        log_callback("$ " + " ".join(command))
    process = subprocess.Popen(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        clean = line.rstrip()
        lines.append(clean)
        if log_callback and clean:
            log_callback(clean)
    if process.wait() != 0:
        raise RuntimeError("Command failed: " + " ".join(command) + "\n" + "\n".join(lines))
    return "\n".join(lines)


class LiveEvaluationRunner(EvaluationRunner):
    def run_yolo_live(self, session_id: str, session_dir: Path, image_path: Path, alias: str, conf: float, visual_style: str, log_callback: Any | None = None) -> dict[str, Any]:
        weight = find_weight(self.weights_dir, alias)
        model_dir = session_dir / alias
        model_dir.mkdir(parents=True, exist_ok=True)
        if weight is None:
            return {"name": alias, "status": "missing weight", "visual": None, "legend": None, "json": None}
        out_json = model_dir / "detections.json"
        project = model_dir / "runs"
        command = [sys.executable, "scripts/infer_yolo.py", "--weights", str(weight), "--source", str(image_path), "--task", "segment", "--conf", str(conf), "--out", str(out_json), "--name", alias, "--project", str(project)]
        if visual_style == "original_labels":
            command.append("--save-visuals")
        if self.yolo_device:
            command += ["--device", self.yolo_device]
        live_run_command(command, self.project_dir, log_callback)
        visual = first_visual(project / alias)
        if visual_style != "original_labels":
            visual = render_detection_visual(image_path, out_json, model_dir / f"{alias}_{visual_style}.jpg", visual_style) or visual
        elif visual:
            shutil.copy2(visual, model_dir / visual.name)
            visual = model_dir / visual.name
        legend = render_legend_image(out_json, model_dir / f"{alias}_renk_cetveli.jpg", alias)
        insert_artifact(self.db_path, session_id, "json", f"{alias}_json", out_json, {"weight": str(weight), "conf": conf})
        if visual:
            insert_artifact(self.db_path, session_id, "image", f"{alias}_visual", visual)
        if legend:
            insert_artifact(self.db_path, session_id, "image", f"{alias}_legend", legend)
        return {"name": alias, "status": "ok", "visual": visual, "legend": legend, "json": out_json}

    def run_hybrid_live(self, session_id: str, session_dir: Path, image_path: Path, visual_style: str, log_callback: Any | None = None) -> dict[str, Any]:
        hybrid_dir = session_dir / "hybrid_yolo_sam2"
        hybrid_dir.mkdir(parents=True, exist_ok=True)
        raw = self.run_yolo_live(session_id, session_dir, image_path, "yolo_v3_no_erasing", 0.25, visual_style, log_callback)
        raw_json = raw.get("json")
        if not raw_json:
            return {"name": "hybrid_yolo_sam2", "status": "missing yolo_v3", "visual": None, "legend": None, "json": None}
        filtered_json = hybrid_dir / "yolo_filtered_conf078.json"
        live_run_command([sys.executable, "scripts/filter_detections.py", "--input", str(raw_json), "--out", str(filtered_json), "--default-conf", "0.78", "--class-threshold", "cam=0.82", "--class-threshold", "ahsap_dograma=0.78", "--class-threshold", "camur_harc=0.72", "--min-area-ratio", "0.00005", "--max-area-ratio", "0.35", "--nms-iou", "0.25"], self.project_dir, log_callback)
        legend = render_legend_image(filtered_json, hybrid_dir / "hybrid_yolo_sam2_renk_cetveli.jpg", "hybrid_yolo_sam2")
        if not self.sam2_dir or not self.sam2_dir.exists() or not self.sam2_checkpoint or not self.sam2_checkpoint.exists():
            return {"name": "hybrid_yolo_sam2", "status": "filtered only; SAM2 not installed", "visual": raw.get("visual"), "legend": legend, "json": filtered_json, "raw_json": raw_json}
        hybrid_json = hybrid_dir / "hybrid_yolo_sam2.json"
        visual_dir = hybrid_dir / "visuals"
        live_run_command([sys.executable, str(self.project_dir / "scripts/refine_with_sam2.py"), "--detections", str(filtered_json), "--checkpoint", str(self.sam2_checkpoint), "--model-cfg", self.sam2_model_cfg, "--out", str(hybrid_json), "--visual-dir", str(visual_dir), "--device", self.sam2_device or "cuda"], self.sam2_dir, log_callback)
        visual = first_visual(visual_dir)
        legend = render_legend_image(hybrid_json, hybrid_dir / "hybrid_yolo_sam2_renk_cetveli.jpg", "hybrid_yolo_sam2") or legend
        if visual:
            insert_artifact(self.db_path, session_id, "image", "hybrid_yolo_sam2_visual", visual)
        insert_artifact(self.db_path, session_id, "json", "hybrid_yolo_sam2_json", hybrid_json)
        return {"name": "hybrid_yolo_sam2", "status": "ok", "visual": visual, "legend": legend, "json": hybrid_json, "raw_json": raw_json}

    def run_session(self, image_file: str, api_key: str, vlm_model: str, run_v1: bool, run_v2: bool, run_v3: bool, run_hybrid: bool, vlm_image_only: bool, vlm_yolo: bool, vlm_sam: bool, vlm_hybrid: bool, visual_style: str, log_callback: Any | None = None):
        debug: list[str] = []
        def emit(message: str) -> None:
            debug.append(message)
            if log_callback:
                log_callback(message)
        session_id = f"session_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        session_dir = self.reports_dir / "evaluation_sessions" / session_id
        image_path = normalize_input_image(image_file, session_dir / "input")
        insert_session(self.db_path, session_id, image_path, session_dir)
        insert_artifact(self.db_path, session_id, "image", "input", image_path)
        emit(f"Session: {session_id}")
        results = []
        if run_v1:
            emit("YOLO v1 basliyor...")
            results.append(self.run_yolo_live(session_id, session_dir, image_path, "yolo_v1_default", 0.25, visual_style, emit))
        if run_v2:
            emit("YOLO v2 basliyor...")
            results.append(self.run_yolo_live(session_id, session_dir, image_path, "yolo_v2_aug_controlled", 0.25, visual_style, emit))
        if run_v3:
            emit("YOLO v3 basliyor...")
            results.append(self.run_yolo_live(session_id, session_dir, image_path, "yolo_v3_no_erasing", 0.25, visual_style, emit))
        hybrid_result = None
        if run_hybrid or vlm_sam or vlm_hybrid:
            emit("Hybrid YOLO+SAM2 basliyor...")
            hybrid_result = self.run_hybrid_live(session_id, session_dir, image_path, visual_style, emit)
            if run_hybrid:
                results.append(hybrid_result)
        report_parts = ["# Cephe Lejant Raporu", "", f"Session: `{session_id}`", "", "## Kullanilan Yontemler", "- YOLO/SAM2 ile cephe elemani tespiti ve maske/kutu ciktisi uretildi.", "- VLM raporlari bina ve cephe yorumu icin kullanildi; rapor metinleri algoritma anlatimi yerine yapinin mimari okumasina odaklanir.", "", "----", ""]
        gallery = [(str(image_path), "input")]
        targets, details = [], []
        for result in results:
            name = result["name"]
            section = [f"## Tespit Ozeti: {name}", f"Durum: {result['status']}"]
            if result.get("json"):
                section.append(f"Tespit ozeti: `{json.dumps(detection_counts(one_record(result['json'])), ensure_ascii=False)}`")
            report_parts += section + ["", "----", ""]
            for key, suffix in (("visual", ""), ("legend", "_renk_cetveli")):
                if result.get(key):
                    gallery.append((str(result[key]), name + suffix))
            targets.append(name)
            details.append({"target": name, "title": title_for(name), "image": str(result.get("visual") or ""), "legend": str(result.get("legend") or ""), "content": "\n".join(section)})
        if any([vlm_image_only, vlm_yolo, vlm_sam, vlm_hybrid]):
            vlm_dir = session_dir / "vlm_reports"
            vlm_dir.mkdir(parents=True, exist_ok=True)
            assisted_payload = make_report_payload(raw_json=hybrid_result.get("raw_json") if hybrid_result else None, filtered_json=(session_dir / "hybrid_yolo_sam2" / "yolo_filtered_conf078.json"), hybrid_json=hybrid_result.get("json") if hybrid_result else None)
            yolo_payload = make_yolo_payload(results)
            jobs = []
            if vlm_image_only:
                jobs.append(("vlm_image_only", "Sadece Resim ile LLM Cephe Raporu", "Bu gorsel icin bina/cephe odakli mimari cephe lejant raporu yaz. Model veya algoritma anlatma. Cephede gorulen elemanlari, malzemeleri, cephe duzenini ve koruma/restorasyon acisindan dikkat ceken noktalari yorumla."))
            if vlm_yolo:
                jobs.append(("vlm_yolo_assisted", "YOLO Tespit Ozeti ile LLM Cephe Raporu", "Bu gorsel ve YOLO tespit ozeti ile bina/cephe odakli mimari cephe lejant raporu yaz. Algoritma karsilastirmasi yapma.\n\n" + json.dumps(yolo_payload, ensure_ascii=False, indent=2)))
            if vlm_sam:
                jobs.append(("vlm_sam2_assisted", "SAM2 Rafine Cikti ile LLM Cephe Raporu", "Bu gorsel ve SAM2 ile rafine edilmis tespit ozeti ile bina/cephe odakli mimari cephe lejant raporu yaz.\n\n" + json.dumps({"sam2_refined": assisted_payload.get("hybrid_yolo_sam2")}, ensure_ascii=False, indent=2)))
            if vlm_hybrid:
                jobs.append(("vlm_hybrid_yolo_sam2", "Hibrit YOLO + SAM2 Cikti ile LLM Cephe Raporu", "Bu gorsel ve hibrit YOLO+SAM2 tespit ozeti ile bina/cephe odakli mimari cephe lejant raporu yaz.\n\n" + json.dumps(assisted_payload, ensure_ascii=False, indent=2)))
            for target, heading, prompt in jobs:
                emit(f"{target} uretiliyor...")
                text = call_vlm(image_path, prompt, api_key=api_key, model=vlm_model)
                save_vlm_report(self.db_path, session_id, vlm_dir, target, text)
                report_parts += [f"## {heading}", "", text, "", "----", ""]
                targets.append(target)
                details.append({"target": target, "title": title_for(target), "image": str(image_path), "legend": "", "content": text})
        visual_summary = make_visual_summary(gallery, session_dir / "visual_summary.jpg")
        if visual_summary:
            gallery.insert(0, (str(visual_summary), "toplu_gorsel_sonuc"))
        targets.append("overall")
        details.append({"target": "overall", "title": title_for("overall"), "image": str(visual_summary or image_path), "legend": "", "content": "Genel degerlendirme ve kullanici yorumu icin ayrilan bolum."})
        summary_path = session_dir / "session_summary.md"
        summary_path.write_text("\n".join(report_parts), encoding="utf-8")
        combined = write_combined_report(session_dir, session_id, visual_summary, details)
        insert_artifact(self.db_path, session_id, "report", "session_summary", summary_path)
        insert_artifact(self.db_path, session_id, "report", "combined_report", combined)
        emit(f"Combined report: {combined}")
        return session_id, gallery, "\n".join(report_parts), str(self.db_path), targets, details, "\n".join(debug)


def build_app(runner: EvaluationRunner):
    import gradio as gr
    auto_api_key, api_key_status = get_openai_api_key()
    feedback_slots = 9
    star_choices = [("No feedback", 0), ("⭐☆☆☆☆", 1), ("⭐⭐☆☆☆", 2), ("⭐⭐⭐☆☆", 3), ("⭐⭐⭐⭐☆", 4), ("⭐⭐⭐⭐⭐", 5)]

    def tab_updates(details):
        ordered = list(details or [])
        updates = []
        for index in range(feedback_slots):
            if index < len(ordered):
                detail = ordered[index]
                updates += [gr.update(label=detail.get("title", "Sonuc")[:28], visible=True), gr.update(value=f"### {detail.get('title')}\n`{detail.get('target')}`", visible=True), gr.update(value=detail.get("image") or None, visible=bool(detail.get("image"))), gr.update(value=detail.get("content", ""), visible=True), gr.update(value=0, visible=True), gr.update(value="", visible=True)]
            else:
                updates += [gr.update(visible=False), gr.update(value="", visible=False), gr.update(value=None, visible=False), gr.update(value="", visible=False), gr.update(value=0, visible=False), gr.update(value="", visible=False)]
        return updates

    def llm_card_updates(details):
        llm = [item for item in list(details or []) if item.get("target", "").startswith("vlm_")]
        updates = []
        for index in range(4):
            if index < len(llm):
                detail = llm[index]
                updates += [gr.update(visible=True), gr.update(value=f"### {detail.get('title')}\n`{detail.get('target')}`\n\n{detail.get('content', '')}", visible=True), gr.update(value=0, visible=True), gr.update(value="", visible=True)]
            else:
                updates += [gr.update(visible=False), gr.update(value="", visible=False), gr.update(value=0, visible=False), gr.update(value="", visible=False)]
        return updates

    def compare_choices(details, kind):
        if kind == "llm":
            filtered = [item for item in list(details or []) if item.get("target", "").startswith("vlm_")]
        else:
            filtered = [item for item in list(details or []) if item.get("image") and not item.get("target", "").startswith("vlm_") and item.get("target") != "overall"]
        return [(item.get("title", item.get("target", "")), item.get("target", "")) for item in filtered]

    def compare_slot_updates(details, kind):
        choices = compare_choices(details, kind)
        defaults = [value for _, value in choices[:4]]
        return [gr.update(choices=choices, value=defaults[index] if index < len(defaults) else None, visible=bool(choices)) for index in range(4)]

    def compare_selected(details, slot_1, slot_2, slot_3, slot_4):
        selected = [slot for slot in [slot_1, slot_2, slot_3, slot_4] if slot]
        by_target = {item.get("target", ""): item for item in list(details or [])}
        updates = []
        for index in range(4):
            if index < len(selected):
                detail = by_target.get(selected[index], {})
                updates += [gr.update(visible=True), gr.update(value=f"### {detail.get('title', selected[index])}\n`{selected[index]}`", visible=True), gr.update(value=detail.get("image") or None, visible=bool(detail.get("image"))), gr.update(value=detail.get("content", ""), visible=True)]
            else:
                updates += [gr.update(visible=False), gr.update(value="", visible=False), gr.update(value=None, visible=False), gr.update(value="", visible=False)]
        return updates

    def run_clicked(image_file, manual_api_key, vlm_model, run_v1, run_v2, run_v3, run_hybrid, vlm_image_only, vlm_yolo, vlm_sam, vlm_hybrid, visual_style):
        if image_file is None:
            raise gr.Error("Once bir gorsel yukle.")
        log_queue: queue.Queue[str] = queue.Queue()
        result_box: dict[str, object] = {}
        def worker():
            try:
                result_box["value"] = runner.run_session(image_file, (manual_api_key or "").strip() or auto_api_key, vlm_model or "gpt-4o", run_v1, run_v2, run_v3, run_hybrid, vlm_image_only, vlm_yolo, vlm_sam, vlm_hybrid, visual_style or "mask_only", log_queue.put)
            except Exception as exc:
                result_box["error"] = f"{exc}\n\n{traceback.format_exc()}"
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        logs = ["Calisma basladi. Terminal akisindan sureci izleyebilirsin."]
        while thread.is_alive():
            while not log_queue.empty():
                logs.append(log_queue.get_nowait())
            yield ("", [], "## Calisiyor\n\nModel ve rapor uretimi devam ediyor.", str(runner.db_path), "\n".join(logs[-250:]), [], [], *tab_updates([]), *llm_card_updates([]), *compare_slot_updates([], "image"), *compare_slot_updates([], "llm"))
            time.sleep(0.5)
        while not log_queue.empty():
            logs.append(log_queue.get_nowait())
        if "error" in result_box:
            err = f"## Hata\n\n```text\n{result_box['error']}\n```"
            yield ("", [], err, str(runner.db_path), "\n".join(logs[-300:]), [], [], *tab_updates([]), *llm_card_updates([]), *compare_slot_updates([], "image"), *compare_slot_updates([], "llm"))
            return
        session_id, gallery, report, db_path, targets, details, debug = result_box["value"]  # type: ignore[misc]
        yield (session_id, gallery, report, db_path, "\n".join((logs + str(debug).splitlines())[-300:]), targets, details, *tab_updates(details), *llm_card_updates(details), *compare_slot_updates(details, "image"), *compare_slot_updates(details, "llm"))

    def save_feedback(session_id, targets, details, *values):
        if not session_id:
            raise gr.Error("Once bir test oturumu calistir.")
        feedback_map: dict[str, dict[str, Any]] = {}
        lines = []
        target_list = list(targets or [])
        for index, target in enumerate(target_list[:feedback_slots]):
            score = values[index * 2]
            comment = (values[index * 2 + 1] or "").strip()
            numeric = float(score) if score and float(score) > 0 else None
            stored = comment or ("no feedback" if numeric is None else "no comment")
            insert_rating(runner.db_path, session_id, target, numeric, stored)
            feedback_map[target] = {"score": numeric, "comment": stored}
            lines.append(f"- **{title_for(target)}** (`{target}`): puan=`{numeric if numeric is not None else 'no feedback'}`; yorum={stored}")
        llm_targets = [item.get("target", "") for item in list(details or []) if item.get("target", "").startswith("vlm_")][:4]
        offset = feedback_slots * 2
        for index, target in enumerate(llm_targets):
            score = values[offset + index * 2]
            comment = (values[offset + index * 2 + 1] or "").strip()
            if not (score and float(score) > 0) and not comment:
                continue
            numeric = float(score) if score and float(score) > 0 else None
            stored = comment or ("no feedback" if numeric is None else "no comment")
            insert_rating(runner.db_path, session_id, target, numeric, stored)
            feedback_map[target] = {"score": numeric, "comment": stored}
        session_dir = runner.reports_dir / "evaluation_sessions" / session_id
        feedback_path = session_dir / "feedback_summary.md"
        feedback_path.write_text("# Puan ve Yorum Ozeti\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
        combined = write_combined_report(session_dir, session_id, session_dir / "visual_summary.jpg", list(details or []), feedback_map)
        insert_artifact(runner.db_path, session_id, "report", "feedback_summary", feedback_path)
        insert_artifact(runner.db_path, session_id, "report", "combined_report_with_feedback", combined)
        return f"Kaydedildi.\nFeedback: {feedback_path}\nBirlesik rapor: {combined}\nDB: {runner.db_path}"

    with gr.Blocks(title="Lejanter Evaluation Interface") as app:
        gr.Markdown("# Lejanter Evaluation Interface")
        gr.Markdown("Tum deneyler varsayilan olarak acik. Gorseller yazisiz uretilir; renk cetveli ayri gorsel olarak kaydedilir.")
        session_state = gr.State("")
        targets_state = gr.State([])
        details_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="Test image", type="filepath")
                gr.Markdown(f"**OpenAI API key:** {api_key_status}")
                manual_api_key = gr.Textbox(label="OpenAI API key manuel yedek", type="password", placeholder="Bos birak: Colab Secrets OPENAI_API_KEY kullanilir")
                vlm_model = gr.Dropdown(label="VLM model", choices=VLM_MODELS, value="gpt-4o", allow_custom_value=True)
                gr.Markdown("### Model ciktilari")
                run_v1 = gr.Checkbox(label="YOLO v1", value=True)
                run_v2 = gr.Checkbox(label="YOLO v2", value=True)
                run_v3 = gr.Checkbox(label="YOLO v3", value=True)
                run_hybrid = gr.Checkbox(label="SAM2 / Hybrid YOLO + SAM2", value=True)
                visual_style = gr.Dropdown(label="Gorsel cizim modu", choices=[("Segmentasyon, yazisiz", "mask_only"), ("Sadece kare/kutu, yazisiz", "box_only"), ("Kucuk etiketler", "small_labels"), ("Orijinal buyuk YOLO etiketleri", "original_labels")], value="mask_only")
                gr.Markdown("### VLM rapor tipleri")
                vlm_image_only = gr.Checkbox(label="Sadece resmi VLM'e ver", value=True)
                vlm_yolo = gr.Checkbox(label="Resim + YOLO sonuclarini VLM'e ver", value=True)
                vlm_sam = gr.Checkbox(label="Resim + SAM2 sonucunu VLM'e ver", value=True)
                vlm_hybrid = gr.Checkbox(label="Resim + YOLO + SAM2 hibrit sonucu VLM'e ver", value=True)
                run_button = gr.Button("Run evaluation", variant="primary")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Result visuals", columns=2, height=520)
                with gr.Accordion("Ham rapor onizleme", open=False):
                    report = gr.Markdown(label="Reports")
                db_path = gr.Textbox(label="SQLite DB path")
                debug_log = gr.Textbox(label="Terminal / Debug log", lines=18, autoscroll=True)

        gr.Markdown("## Sonuc Sekmeleri, Yildiz Puani ve Yorum")
        feedback_components = []
        with gr.Tabs():
            for index in range(feedback_slots):
                with gr.Tab(f"Sonuc {index + 1}", visible=False) as tab:
                    label = gr.Markdown()
                    out_img = gr.Image(label="Cikti gorseli", type="filepath", height=360)
                    content = gr.Markdown()
                    score = gr.Radio(label="Yildiz puani", choices=star_choices, value=0)
                    comment = gr.Textbox(label="Yorum", lines=3)
                    feedback_components.extend([tab, label, out_img, content, score, comment])
        save_button = gr.Button("Tum puan/yorumlari kaydet")
        save_status = gr.Textbox(label="Save status")

        gr.Markdown("## LLM Rapor Karsilastirma")
        llm_cards = []
        with gr.Row():
            for _ in range(4):
                with gr.Column(scale=1, visible=False) as group:
                    card = gr.Markdown()
                    score = gr.Radio(label="Yildiz puani", choices=star_choices, value=0)
                    comment = gr.Textbox(label="Yorum", lines=3)
                    llm_cards.extend([group, card, score, comment])

        gr.Markdown("## Gorsel Sonuclari Karsilastir")
        gr.Markdown("Model/SAM ciktilarini ayri karsilastir. Kart siralamasini dropdown slotlariyla degistirebilirsin.")
        image_slots = []
        with gr.Row():
            for index in range(4):
                image_slots.append(gr.Dropdown(label=f"Gorsel kart {index + 1}", choices=[], value=None, visible=False))
        image_button = gr.Button("Gorselleri karsilastir")
        image_components = []
        for _ in range(2):
            with gr.Row():
                for _ in range(2):
                    with gr.Column(scale=1, visible=False) as group:
                        label = gr.Markdown()
                        img = gr.Image(label="Gorsel", type="filepath", height=300)
                        md = gr.Markdown()
                        image_components.extend([group, label, img, md])

        gr.Markdown("## LLM Raporlarini Karsilastir")
        gr.Markdown("LLM raporlarini ayri karsilastir. 2 secersen yan yana, 3-4 secersen iki satirli 2x2 duzende gorunur.")
        llm_slots = []
        with gr.Row():
            for index in range(4):
                llm_slots.append(gr.Dropdown(label=f"LLM kart {index + 1}", choices=[], value=None, visible=False))
        llm_button = gr.Button("LLM raporlarini karsilastir")
        llm_components = []
        for _ in range(2):
            with gr.Row():
                for _ in range(2):
                    with gr.Column(scale=1, visible=False) as group:
                        label = gr.Markdown()
                        img = gr.Image(label="Kaynak gorsel", type="filepath", height=220)
                        md = gr.Markdown()
                        llm_components.extend([group, label, img, md])

        run_button.click(run_clicked, inputs=[image, manual_api_key, vlm_model, run_v1, run_v2, run_v3, run_hybrid, vlm_image_only, vlm_yolo, vlm_sam, vlm_hybrid, visual_style], outputs=[session_state, gallery, report, db_path, debug_log, targets_state, details_state, *feedback_components, *llm_cards, *image_slots, *llm_slots])
        image_button.click(compare_selected, inputs=[details_state, *image_slots], outputs=image_components)
        llm_button.click(compare_selected, inputs=[details_state, *llm_slots], outputs=llm_components)
        save_inputs = [session_state, targets_state, details_state]
        for index in range(feedback_slots):
            save_inputs.extend([feedback_components[index * 6 + 4], feedback_components[index * 6 + 5]])
        for index in range(4):
            save_inputs.extend([llm_cards[index * 4 + 2], llm_cards[index * 4 + 3]])
        save_button.click(save_feedback, inputs=save_inputs, outputs=[save_status])
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Gradio evaluation UI with Colab Secrets support.")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--sam2-device", default="cuda")
    parser.add_argument("--sam2-dir", type=Path, default=None)
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = args.db_path or args.reports_dir / "evaluation_sessions" / "ratings.sqlite"
    runner = LiveEvaluationRunner(args.project_dir.resolve(), args.reports_dir.resolve(), args.weights_dir.resolve(), db_path.resolve(), args.yolo_device or args.device, args.sam2_device, args.sam2_dir.resolve() if args.sam2_dir else None, args.sam2_checkpoint.resolve() if args.sam2_checkpoint else None, args.sam2_model_cfg)
    build_app(runner).launch(share=args.share, debug=True, allowed_paths=[str(args.project_dir.resolve()), str(args.reports_dir.resolve()), "/tmp"])


if __name__ == "__main__":
    main()
