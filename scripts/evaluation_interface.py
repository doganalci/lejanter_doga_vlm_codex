from __future__ import annotations

import argparse
import base64
import datetime as dt
import traceback
import json
import shutil
import sqlite3
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

VLM_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"]


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{output}")
    return output


def ensure_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            create table if not exists sessions (
                session_id text primary key,
                created_at text not null,
                input_image_path text not null,
                session_dir text not null,
                notes text
            );

            create table if not exists artifacts (
                id integer primary key autoincrement,
                session_id text not null,
                artifact_type text not null,
                name text not null,
                path text not null,
                metadata_json text,
                created_at text not null
            );

            create table if not exists ratings (
                id integer primary key autoincrement,
                session_id text not null,
                target_name text not null,
                score real,
                comment text,
                created_at text not null
            );
            """
        )


def insert_session(db_path: Path, session_id: str, image_path: Path, session_dir: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            insert or replace into sessions(session_id, created_at, input_image_path, session_dir)
            values (?, ?, ?, ?)
            """,
            (session_id, dt.datetime.now().isoformat(), str(image_path), str(session_dir)),
        )


def insert_artifact(
    db_path: Path,
    session_id: str,
    artifact_type: str,
    name: str,
    path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            insert into artifacts(session_id, artifact_type, name, path, metadata_json, created_at)
            values (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                artifact_type,
                name,
                str(path),
                json.dumps(metadata or {}, ensure_ascii=False),
                dt.datetime.now().isoformat(),
            ),
        )


def insert_rating(db_path: Path, session_id: str, target_name: str, score: float | None, comment: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            insert into ratings(session_id, target_name, score, comment, created_at)
            values (?, ?, ?, ?, ?)
            """,
            (session_id, target_name, score, comment, dt.datetime.now().isoformat()),
        )


def load_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_weight(weights_dir: Path, alias: str) -> Path | None:
    for pattern in WEIGHT_ALIASES[alias]:
        matches = sorted(weights_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def normalize_input_image(source_path: str | Path, out_dir: Path) -> Path:
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


def detection_counts(record: dict[str, Any] | None) -> dict[str, Any]:
    if not record:
        return {"total_detections": 0, "class_counts": {}}
    labels = [item.get("label", "unknown") for item in record.get("detections", [])]
    return {"total_detections": len(labels), "class_counts": dict(Counter(labels))}


def one_record(json_path: Path | None) -> dict[str, Any] | None:
    payload = load_json(json_path)
    if payload is None:
        return None
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    return records[0] if records else None


def make_report_payload(raw_json: Path | None, filtered_json: Path | None, hybrid_json: Path | None) -> dict[str, Any]:
    raw = one_record(raw_json)
    filtered = one_record(filtered_json)
    hybrid = one_record(hybrid_json)
    return {
        "instruction": "Use total_detections and class_counts as authoritative counts. Do not recount samples.",
        "raw_yolo": detection_counts(raw),
        "filtered_yolo": detection_counts(filtered),
        "hybrid_yolo_sam2": detection_counts(hybrid),
    }


def make_yolo_payload(results: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "instruction": "Use total_detections and class_counts as authoritative counts. Do not recount samples.",
        "models": {},
    }
    for result in results:
        if not result["name"].startswith("yolo_") or not result.get("json"):
            continue
        payload["models"][result["name"]] = detection_counts(one_record(result["json"]))
    return payload


def save_vlm_report(
    db_path: Path,
    session_id: str,
    vlm_dir: Path,
    report_name: str,
    content: str,
    payload: dict[str, Any] | None = None,
) -> Path:
    report_path = vlm_dir / f"{report_name}.md"
    report_path.write_text(content, encoding="utf-8")
    insert_artifact(db_path, session_id, "report", report_name, report_path, payload)
    if payload is not None:
        payload_path = vlm_dir / f"{report_name}_payload.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        insert_artifact(db_path, session_id, "json", f"{report_name}_payload", payload_path)
    return report_path


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def call_vlm(
    image_path: Path,
    prompt: str,
    api_key: str,
    model: str,
    base_url: str = "https://api.openai.com/v1",
    temperature: float = 0.2,
) -> str:
    if not api_key.strip():
        return "VLM raporu uretilmedi: API key girilmedi."
    import requests

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": temperature,
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
        return (
            "VLM raporu uretilmedi: OpenAI API key yetkisiz veya hatali gorunuyor. "
            "Anahtar `sk-...` ile baslamali ve bu proje/model icin yetkili olmali."
        )
    if response.status_code == 404:
        return f"VLM raporu uretilmedi: `{model}` modeli bu API key ile bulunamadi veya erisilebilir degil."
    if response.status_code >= 400:
        return f"VLM raporu uretilmedi: HTTP {response.status_code}\n\n{response.text[:1200]}"
    return response.json()["choices"][0]["message"]["content"]


class EvaluationRunner:
    def __init__(
        self,
        project_dir: Path,
        reports_dir: Path,
        weights_dir: Path,
        db_path: Path,
        yolo_device: str,
        sam2_device: str,
        sam2_dir: Path | None,
        sam2_checkpoint: Path | None,
        sam2_model_cfg: str,
    ) -> None:
        self.project_dir = project_dir
        self.reports_dir = reports_dir
        self.weights_dir = weights_dir
        self.db_path = db_path
        self.yolo_device = yolo_device
        self.sam2_device = sam2_device
        self.sam2_dir = sam2_dir
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_model_cfg = sam2_model_cfg
        ensure_db(db_path)

    def run_yolo(self, session_id: str, session_dir: Path, image_path: Path, alias: str, conf: float) -> dict[str, Any]:
        weight = find_weight(self.weights_dir, alias)
        model_dir = session_dir / alias
        model_dir.mkdir(parents=True, exist_ok=True)
        if weight is None:
            return {"name": alias, "status": "missing weight", "visual": None, "json": None}

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
        if self.yolo_device:
            command.extend(["--device", self.yolo_device])
        run_command(command, self.project_dir)

        visual = first_visual(project / run_name)
        if visual:
            shutil.copy2(visual, model_dir / visual.name)
            visual = model_dir / visual.name

        metadata = {"weight": str(weight), "conf": conf}
        insert_artifact(self.db_path, session_id, "json", f"{alias}_json", out_json, metadata)
        if visual:
            insert_artifact(self.db_path, session_id, "image", f"{alias}_visual", visual, metadata)
        return {"name": alias, "status": "ok", "visual": visual, "json": out_json}

    def run_hybrid(self, session_id: str, session_dir: Path, image_path: Path) -> dict[str, Any]:
        hybrid_dir = session_dir / "hybrid_yolo_sam2"
        hybrid_dir.mkdir(parents=True, exist_ok=True)
        raw = self.run_yolo(session_id, session_dir, image_path, "yolo_v3_no_erasing", conf=0.25)
        raw_json = raw.get("json")
        if not raw_json:
            return {"name": "hybrid_yolo_sam2", "status": "missing yolo_v3", "visual": None, "json": None}

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
            self.project_dir,
        )
        insert_artifact(self.db_path, session_id, "json", "hybrid_filtered_json", filtered_json)

        if not self.sam2_dir or not self.sam2_dir.exists() or not self.sam2_checkpoint or not self.sam2_checkpoint.exists():
            return {
                "name": "hybrid_yolo_sam2",
                "status": "filtered only; SAM2 not installed",
                "visual": raw.get("visual"),
                "json": filtered_json,
            }

        hybrid_json = hybrid_dir / "hybrid_yolo_sam2.json"
        visual_dir = hybrid_dir / "visuals"
        run_command(
            [
                sys.executable,
                str(self.project_dir / "scripts/refine_with_sam2.py"),
                "--detections",
                str(filtered_json),
                "--checkpoint",
                str(self.sam2_checkpoint),
                "--model-cfg",
                self.sam2_model_cfg,
                "--out",
                str(hybrid_json),
                "--visual-dir",
                str(visual_dir),
                "--device",
                self.sam2_device or "cuda",
            ],
            self.sam2_dir,
        )
        visual = first_visual(visual_dir)
        insert_artifact(self.db_path, session_id, "json", "hybrid_yolo_sam2_json", hybrid_json)
        if visual:
            insert_artifact(self.db_path, session_id, "image", "hybrid_yolo_sam2_visual", visual)
        return {"name": "hybrid_yolo_sam2", "status": "ok", "visual": visual, "json": hybrid_json, "raw_json": raw_json}

    def run_session(
        self,
        image_file: str,
        api_key: str,
        vlm_model: str,
        run_v1: bool,
        run_v2: bool,
        run_v3: bool,
        run_hybrid: bool,
        vlm_image_only: bool,
        vlm_yolo: bool,
        vlm_sam: bool,
        vlm_hybrid: bool,
    ) -> tuple[str, list[tuple[str, str]], str, str, list[str], str]:
        session_id = f"session_{now_stamp()}_{uuid.uuid4().hex[:8]}"
        session_dir = self.reports_dir / "evaluation_sessions" / session_id
        debug_log = [f"Session: {session_id}", f"Session dir: {session_dir}"]
        input_dir = session_dir / "input"
        image_path = normalize_input_image(image_file, input_dir)
        debug_log.append(f"Input image: {image_path}")
        insert_session(self.db_path, session_id, image_path, session_dir)
        insert_artifact(self.db_path, session_id, "image", "input", image_path)

        results: list[dict[str, Any]] = []
        if run_v1:
            result = self.run_yolo(session_id, session_dir, image_path, "yolo_v1_default", conf=0.25)
            results.append(result)
            debug_log.append(f"YOLO v1: {result['status']} json={result.get('json')} visual={result.get('visual')}")
        if run_v2:
            result = self.run_yolo(session_id, session_dir, image_path, "yolo_v2_aug_controlled", conf=0.25)
            results.append(result)
            debug_log.append(f"YOLO v2: {result['status']} json={result.get('json')} visual={result.get('visual')}")
        if run_v3:
            result = self.run_yolo(session_id, session_dir, image_path, "yolo_v3_no_erasing", conf=0.25)
            results.append(result)
            debug_log.append(f"YOLO v3: {result['status']} json={result.get('json')} visual={result.get('visual')}")
        hybrid_result = None
        if run_hybrid or vlm_sam or vlm_hybrid:
            hybrid_result = self.run_hybrid(session_id, session_dir, image_path)
            debug_log.append(
                f"Hybrid YOLO+SAM2: {hybrid_result['status']} "
                f"json={hybrid_result.get('json')} visual={hybrid_result.get('visual')}"
            )
            if run_hybrid:
                results.append(hybrid_result)

        reports_md = ["# Evaluation Results", "", f"Session: `{session_id}`", ""]
        gallery: list[tuple[str, str]] = [(str(image_path), "input")]
        target_names = ["overall"]
        for result in results:
            reports_md.append(f"## {result['name']}")
            reports_md.append(f"Status: {result['status']}")
            if result.get("json"):
                counts = detection_counts(one_record(result["json"]))
                reports_md.append(f"Detections: `{json.dumps(counts, ensure_ascii=False)}`")
            reports_md.append("")
            if result.get("visual"):
                gallery.append((str(result["visual"]), result["name"]))
            target_names.append(result["name"])

        if any([vlm_image_only, vlm_yolo, vlm_sam, vlm_hybrid]):
            assisted_payload = make_report_payload(
                raw_json=hybrid_result.get("raw_json") if hybrid_result else None,
                filtered_json=(session_dir / "hybrid_yolo_sam2" / "yolo_filtered_conf078.json"),
                hybrid_json=hybrid_result.get("json") if hybrid_result else None,
            )
            yolo_payload = make_yolo_payload(results)
            vlm_dir = session_dir / "vlm_reports"
            vlm_dir.mkdir(parents=True, exist_ok=True)

            if vlm_image_only:
                image_only = call_vlm(
                    image_path,
                    "Bu cephe gorselini mimari cephe lejant raporu olarak yorumla. Sayilari tahminse belirt.",
                    api_key=api_key,
                    model=vlm_model,
                )
                save_vlm_report(self.db_path, session_id, vlm_dir, "vlm_image_only", image_only)
                reports_md.extend(["## VLM - Sadece Gorsel", image_only, ""])
                target_names.append("vlm_image_only")
                debug_log.append("VLM image-only report generated.")

            if vlm_yolo:
                yolo_report = call_vlm(
                    image_path,
                    (
                        "Bu gorsel ve asagidaki YOLO model ciktilarina gore mimari cephe lejant raporu yaz. "
                        "Sayi olarak sadece total_detections ve class_counts alanlarini kullan. "
                        "YOLO versiyonlarini kisa karsilastir.\n\n"
                        + json.dumps(yolo_payload, ensure_ascii=False, indent=2)
                    ),
                    api_key=api_key,
                    model=vlm_model,
                )
                save_vlm_report(self.db_path, session_id, vlm_dir, "vlm_yolo_assisted", yolo_report, yolo_payload)
                reports_md.extend(["## VLM - YOLO Ozetli", yolo_report, ""])
                target_names.append("vlm_yolo_assisted")
                debug_log.append("VLM YOLO-assisted report generated.")

            if vlm_sam:
                sam_payload = {
                    "instruction": "Use hybrid_yolo_sam2 counts as the SAM2-refined mask result. SAM2 is prompted by YOLO boxes.",
                    "sam2_refined": assisted_payload["hybrid_yolo_sam2"],
                }
                sam_report = call_vlm(
                    image_path,
                    (
                        "Bu gorsel ve SAM2 ile iyilestirilmis maske ozetine gore cephe lejant raporu yaz. "
                        "SAM2 sonucunun YOLO kutulari ile yonlendirildigini belirt. "
                        "Sayi olarak sadece total_detections ve class_counts alanlarini kullan.\n\n"
                        + json.dumps(sam_payload, ensure_ascii=False, indent=2)
                    ),
                    api_key=api_key,
                    model=vlm_model,
                )
                save_vlm_report(self.db_path, session_id, vlm_dir, "vlm_sam2_assisted", sam_report, sam_payload)
                reports_md.extend(["## VLM - SAM2 Ozetli", sam_report, ""])
                target_names.append("vlm_sam2_assisted")
                debug_log.append("VLM SAM2-assisted report generated.")

            if vlm_hybrid:
                hybrid_report = call_vlm(
                    image_path,
                    (
                        "Bu gorsel ve asagidaki YOLO + SAM2 hibrit ozetine gore teknik mimari cephe lejant raporu yaz. "
                        "Raw YOLO, filtrelenmis YOLO ve SAM2 ile iyilestirilmis sonucu ayri ayri yorumla. "
                        "Sayi olarak sadece total_detections ve class_counts alanlarini kullan.\n\n"
                        + json.dumps(assisted_payload, ensure_ascii=False, indent=2)
                    ),
                    api_key=api_key,
                    model=vlm_model,
                )
                save_vlm_report(self.db_path, session_id, vlm_dir, "vlm_hybrid_yolo_sam2", hybrid_report, assisted_payload)
                reports_md.extend(["## VLM - YOLO + SAM2 Hibrit", hybrid_report, ""])
                target_names.append("vlm_hybrid_yolo_sam2")
                debug_log.append("VLM hybrid report generated.")

        summary_path = session_dir / "session_summary.md"
        summary_path.write_text("\n".join(reports_md), encoding="utf-8")
        insert_artifact(self.db_path, session_id, "report", "session_summary", summary_path)
        debug_log.append(f"Summary: {summary_path}")
        debug_log.append(f"SQLite DB: {self.db_path}")

        return session_id, gallery, "\n".join(reports_md), str(self.db_path), target_names, "\n".join(debug_log)


def build_app(runner: EvaluationRunner):
    import gradio as gr

    def run_clicked(
        image_file,
        api_key,
        vlm_model,
        run_v1,
        run_v2,
        run_v3,
        run_hybrid,
        vlm_image_only,
        vlm_yolo,
        vlm_sam,
        vlm_hybrid,
    ):
        if image_file is None:
            raise gr.Error("Once bir gorsel yukle.")
        try:
            session_id, gallery, report, db_path, targets, debug_log = runner.run_session(
                image_file=image_file,
                api_key=api_key or "",
                vlm_model=vlm_model or "gpt-4o",
                run_v1=run_v1,
                run_v2=run_v2,
                run_v3=run_v3,
                run_hybrid=run_hybrid,
                vlm_image_only=vlm_image_only,
                vlm_yolo=vlm_yolo,
                vlm_sam=vlm_sam,
                vlm_hybrid=vlm_hybrid,
            )
            return session_id, gallery, report, db_path, debug_log, gr.update(choices=targets, value="overall")
        except Exception as exc:
            error_text = f"## Hata\n\n```text\n{exc}\n\n{traceback.format_exc()}\n```"
            return "", [], error_text, str(runner.db_path), traceback.format_exc(), gr.update(choices=["overall"], value="overall")

    def save_rating_clicked(session_id, target_name, score, comment):
        if not session_id:
            raise gr.Error("Once bir test oturumu calistir.")
        insert_rating(runner.db_path, session_id, target_name or "overall", score, comment or "")
        return f"Saved rating for {target_name or 'overall'} in {runner.db_path}"

    with gr.Blocks(title="Lejanter Evaluation Interface") as app:
        gr.Markdown("# Lejanter Evaluation Interface")
        gr.Markdown("Resim yukle, modelleri calistir, raporlari gor, puan ve yorumlari Drive SQLite DB'ye kaydet.")
        session_state = gr.State("")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="Test image", type="filepath")
                api_key = gr.Textbox(label="OpenAI API key", type="password")
                vlm_model = gr.Dropdown(label="VLM model", choices=VLM_MODELS, value="gpt-4o", allow_custom_value=True)
                gr.Markdown("### Model ciktilari")
                run_v1 = gr.Checkbox(label="YOLO v1", value=True)
                run_v2 = gr.Checkbox(label="YOLO v2", value=True)
                run_v3 = gr.Checkbox(label="YOLO v3", value=True)
                run_hybrid = gr.Checkbox(label="SAM2 / Hybrid YOLO + SAM2", value=True)
                gr.Markdown("### VLM rapor tipleri")
                vlm_image_only = gr.Checkbox(label="Sadece resmi VLM'e ver", value=False)
                vlm_yolo = gr.Checkbox(label="Resim + YOLO sonuclarini VLM'e ver", value=False)
                vlm_sam = gr.Checkbox(label="Resim + SAM2 sonucunu VLM'e ver", value=False)
                vlm_hybrid = gr.Checkbox(label="Resim + YOLO + SAM2 hibrit sonucu VLM'e ver", value=False)
                run_button = gr.Button("Run evaluation", variant="primary")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Result visuals", columns=2, height=520)
                report = gr.Markdown(label="Reports")
                db_path = gr.Textbox(label="SQLite DB path")
                debug_log = gr.Textbox(label="Debug log", lines=12)

        gr.Markdown("## Rating")
        with gr.Row():
            target = gr.Dropdown(label="Target", choices=["overall"], value="overall")
            score = gr.Slider(label="Score", minimum=1, maximum=5, step=1, value=3)
        comment = gr.Textbox(label="Comment", lines=4)
        save_button = gr.Button("Save rating/comment")
        save_status = gr.Textbox(label="Save status")

        run_button.click(
            run_clicked,
            inputs=[
                image,
                api_key,
                vlm_model,
                run_v1,
                run_v2,
                run_v3,
                run_hybrid,
                vlm_image_only,
                vlm_yolo,
                vlm_sam,
                vlm_hybrid,
            ],
            outputs=[session_state, gallery, report, db_path, debug_log, target],
        )
        save_button.click(save_rating_clicked, inputs=[session_state, target, score, comment], outputs=[save_status])
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Gradio evaluation UI.")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--device", default="0", help="Backward-compatible alias for --yolo-device.")
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
    runner = EvaluationRunner(
        project_dir=args.project_dir.resolve(),
        reports_dir=args.reports_dir.resolve(),
        weights_dir=args.weights_dir.resolve(),
        db_path=db_path.resolve(),
        yolo_device=args.yolo_device or args.device,
        sam2_device=args.sam2_device,
        sam2_dir=args.sam2_dir.resolve() if args.sam2_dir else None,
        sam2_checkpoint=args.sam2_checkpoint.resolve() if args.sam2_checkpoint else None,
        sam2_model_cfg=args.sam2_model_cfg,
    )
    app = build_app(runner)
    app.launch(
        share=args.share,
        debug=True,
        allowed_paths=[str(args.project_dir.resolve()), str(args.reports_dir.resolve()), "/tmp"],
    )


if __name__ == "__main__":
    main()
