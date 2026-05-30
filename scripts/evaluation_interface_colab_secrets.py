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
    command_text = " ".join(command)
    if log_callback:
        log_callback(f"$ {command_text}")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        clean = line.rstrip()
        lines.append(clean)
        if log_callback and clean:
            log_callback(clean)
    return_code = process.wait()
    output = "\n".join(lines)
    if return_code != 0:
        raise RuntimeError(f"Command failed: {command_text}\n{output}")
    return output


class LiveEvaluationRunner(EvaluationRunner):
    def run_yolo_live(
        self,
        session_id: str,
        session_dir: Path,
        image_path: Path,
        alias: str,
        conf: float,
        log_callback: Any | None = None,
    ) -> dict[str, Any]:
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
        live_run_command(command, self.project_dir, log_callback=log_callback)
        visual = first_visual(project / run_name)
        if visual:
            shutil.copy2(visual, model_dir / visual.name)
            visual = model_dir / visual.name
        metadata = {"weight": str(weight), "conf": conf}
        insert_artifact(self.db_path, session_id, "json", f"{alias}_json", out_json, metadata)
        if visual:
            insert_artifact(self.db_path, session_id, "image", f"{alias}_visual", visual, metadata)
        return {"name": alias, "status": "ok", "visual": visual, "json": out_json}

    def run_hybrid_live(
        self,
        session_id: str,
        session_dir: Path,
        image_path: Path,
        log_callback: Any | None = None,
    ) -> dict[str, Any]:
        hybrid_dir = session_dir / "hybrid_yolo_sam2"
        hybrid_dir.mkdir(parents=True, exist_ok=True)
        raw = self.run_yolo_live(session_id, session_dir, image_path, "yolo_v3_no_erasing", conf=0.25, log_callback=log_callback)
        raw_json = raw.get("json")
        if not raw_json:
            return {"name": "hybrid_yolo_sam2", "status": "missing yolo_v3", "visual": None, "json": None}
        filtered_json = hybrid_dir / "yolo_filtered_conf078.json"
        live_run_command(
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
            log_callback=log_callback,
        )
        insert_artifact(self.db_path, session_id, "json", "hybrid_filtered_json", filtered_json)
        if not self.sam2_dir or not self.sam2_dir.exists() or not self.sam2_checkpoint or not self.sam2_checkpoint.exists():
            return {"name": "hybrid_yolo_sam2", "status": "filtered only; SAM2 not installed", "visual": raw.get("visual"), "json": filtered_json}
        hybrid_json = hybrid_dir / "hybrid_yolo_sam2.json"
        visual_dir = hybrid_dir / "visuals"
        live_run_command(
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
            log_callback=log_callback,
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
        log_callback: Any | None = None,
    ) -> tuple[str, list[tuple[str, str]], str, str, list[str], str]:
        debug_log: list[str] = []

        def emit(message: str) -> None:
            debug_log.append(message)
            if log_callback:
                log_callback(message)

        session_id = f"session_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        session_dir = self.reports_dir / "evaluation_sessions" / session_id
        emit(f"Session: {session_id}")
        emit(f"Session dir: {session_dir}")
        image_path = normalize_input_image(image_file, session_dir / "input")
        emit(f"Input image: {image_path}")
        insert_session(self.db_path, session_id, image_path, session_dir)
        insert_artifact(self.db_path, session_id, "image", "input", image_path)

        results: list[dict[str, Any]] = []
        if run_v1:
            emit("YOLO v1 basliyor...")
            result = self.run_yolo_live(session_id, session_dir, image_path, "yolo_v1_default", conf=0.25, log_callback=emit)
            results.append(result)
            emit(f"YOLO v1: {result['status']} json={result.get('json')} visual={result.get('visual')}")
        if run_v2:
            emit("YOLO v2 basliyor...")
            result = self.run_yolo_live(session_id, session_dir, image_path, "yolo_v2_aug_controlled", conf=0.25, log_callback=emit)
            results.append(result)
            emit(f"YOLO v2: {result['status']} json={result.get('json')} visual={result.get('visual')}")
        if run_v3:
            emit("YOLO v3 basliyor...")
            result = self.run_yolo_live(session_id, session_dir, image_path, "yolo_v3_no_erasing", conf=0.25, log_callback=emit)
            results.append(result)
            emit(f"YOLO v3: {result['status']} json={result.get('json')} visual={result.get('visual')}")

        hybrid_result = None
        if run_hybrid or vlm_sam or vlm_hybrid:
            emit("Hybrid YOLO+SAM2 basliyor...")
            hybrid_result = self.run_hybrid_live(session_id, session_dir, image_path, log_callback=emit)
            emit(f"Hybrid YOLO+SAM2: {hybrid_result['status']} json={hybrid_result.get('json')} visual={hybrid_result.get('visual')}")
            if run_hybrid:
                results.append(hybrid_result)

        reports_md = [
            "# Cephe Lejant Raporu",
            "",
            f"Session: `{session_id}`",
            "",
            "## Kullanilan Yontemler",
            "- YOLO segmentasyon modelleri ile cephe elemanlari tespit edildi.",
            "- SAM2, YOLO tespitlerinden gelen maskeleri gorsel olarak rafine etmek icin kullanildi.",
            "- LLM/VLM, model ciktilarini mimari cephe lejant raporuna donusturmek icin kullanildi.",
            "- Sayisal adetlerde model JSON ciktilari esas alindi; LLM'den yeni sayi uydurmasi istenmedi.",
            "",
            "----",
            "",
        ]
        gallery: list[tuple[str, str]] = [(str(image_path), "input")]
        target_names: list[str] = []
        for result in results:
            reports_md.append(f"## Tespit Ozeti: {result['name']}")
            reports_md.append(f"Durum: {result['status']}")
            if result.get("json"):
                counts = detection_counts(one_record(result["json"]))
                reports_md.append(f"Tespit ozeti: `{json.dumps(counts, ensure_ascii=False)}`")
            reports_md.extend(["", "----", ""])
            if result.get("visual"):
                gallery.append((str(result["visual"]), result["name"]))
            target_names.append(result["name"])

        if any([vlm_image_only, vlm_yolo, vlm_sam, vlm_hybrid]):
            emit("VLM raporlama basliyor...")
            assisted_payload = make_report_payload(
                raw_json=hybrid_result.get("raw_json") if hybrid_result else None,
                filtered_json=(session_dir / "hybrid_yolo_sam2" / "yolo_filtered_conf078.json"),
                hybrid_json=hybrid_result.get("json") if hybrid_result else None,
            )
            yolo_payload = make_yolo_payload(results)
            vlm_dir = session_dir / "vlm_reports"
            vlm_dir.mkdir(parents=True, exist_ok=True)
            if vlm_image_only:
                emit("VLM image-only raporu uretiliyor...")
                image_only = call_vlm(
                    image_path,
                    "Bu gorsel icin bina/cephe odakli mimari cephe lejant raporu yaz. Model veya algoritma anlatma. Cephede gorulen elemanlari, malzemeleri, cephe duzenini, olasi koruma/restorasyon acisindan dikkat ceken noktalari yorumla. Sayilar tahminse acikca tahmini oldugunu belirt.",
                    api_key=api_key,
                    model=vlm_model,
                )
                save_vlm_report(self.db_path, session_id, vlm_dir, "vlm_image_only", image_only)
                reports_md.extend(["## Sadece Resim ile LLM Cephe Raporu", image_only, "", "----", ""])
                target_names.append("vlm_image_only")
            if vlm_yolo:
                emit("VLM YOLO destekli rapor uretiliyor...")
                yolo_report = call_vlm(
                    image_path,
                    "Bu gorsel ve asagidaki tespit ozeti ile bina/cephe odakli mimari cephe lejant raporu yaz. Algoritma karsilastirmasi yapma; yalnizca raporun basinda 'YOLO tespit ozeti kullanildi' diye kisa belirt. Sayi olarak sadece total_detections ve class_counts alanlarini kullan. Cephe elemanlarini, malzeme izlenimlerini, cephe duzenini ve koruma/restorasyon acisindan yorumlari acikla.\n\n"
                    + json.dumps(yolo_payload, ensure_ascii=False, indent=2),
                    api_key=api_key,
                    model=vlm_model,
                )
                save_vlm_report(self.db_path, session_id, vlm_dir, "vlm_yolo_assisted", yolo_report, yolo_payload)
                reports_md.extend(["## YOLO Tespit Ozeti ile LLM Cephe Raporu", yolo_report, "", "----", ""])
                target_names.append("vlm_yolo_assisted")
            if vlm_sam:
                emit("VLM SAM2 destekli rapor uretiliyor...")
                sam_payload = {"instruction": "Use hybrid_yolo_sam2 counts as the SAM2-refined mask result. SAM2 is prompted by YOLO boxes.", "sam2_refined": assisted_payload["hybrid_yolo_sam2"]}
                sam_report = call_vlm(
                    image_path,
                    "Bu gorsel ve asagidaki rafine tespit ozeti ile bina/cephe odakli mimari cephe lejant raporu yaz. Algoritma detaylarina girme; yalnizca raporun basinda 'SAM2 ile rafine edilmis tespit ozeti kullanildi' diye kisa belirt. Sayi olarak sadece total_detections ve class_counts alanlarini kullan. Cephedeki elemanlari, malzemeleri ve cephe karakterini yorumla.\n\n"
                    + json.dumps(sam_payload, ensure_ascii=False, indent=2),
                    api_key=api_key,
                    model=vlm_model,
                )
                save_vlm_report(self.db_path, session_id, vlm_dir, "vlm_sam2_assisted", sam_report, sam_payload)
                reports_md.extend(["## SAM2 Rafine Cikti ile LLM Cephe Raporu", sam_report, "", "----", ""])
                target_names.append("vlm_sam2_assisted")
            if vlm_hybrid:
                emit("VLM hybrid raporu uretiliyor...")
                hybrid_report = call_vlm(
                    image_path,
                    "Bu gorsel ve asagidaki hibrit tespit ozeti ile bina/cephe odakli mimari cephe lejant raporu yaz. Algoritma ayrintilarina girme; yalnizca raporun basinda 'hibrit YOLO+SAM2 tespit ozeti kullanildi' diye kisa belirt. Sayi olarak sadece total_detections ve class_counts alanlarini kullan. Cephe elemanlari, malzeme karakteri, mimari duzen ve olasi koruma/restorasyon degerlendirmesini anlat.\n\n"
                    + json.dumps(assisted_payload, ensure_ascii=False, indent=2),
                    api_key=api_key,
                    model=vlm_model,
                )
                save_vlm_report(self.db_path, session_id, vlm_dir, "vlm_hybrid_yolo_sam2", hybrid_report, assisted_payload)
                reports_md.extend(["## Hibrit YOLO + SAM2 Cikti ile LLM Cephe Raporu", hybrid_report, "", "----", ""])
                target_names.append("vlm_hybrid_yolo_sam2")

        target_names.append("overall")
        summary_path = session_dir / "session_summary.md"
        summary_path.write_text("\n".join(reports_md), encoding="utf-8")
        insert_artifact(self.db_path, session_id, "report", "session_summary", summary_path)
        emit(f"Summary: {summary_path}")
        emit(f"SQLite DB: {self.db_path}")
        return session_id, gallery, "\n".join(reports_md), str(self.db_path), target_names, "\n".join(debug_log)


def build_app(runner: EvaluationRunner):
    import gradio as gr

    auto_api_key, api_key_status = get_openai_api_key()
    feedback_slots = 9

    def feedback_title(target_name: str) -> str:
        titles = {
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
        return titles.get(target_name, target_name)

    def feedback_updates(targets: list[str] | None):
        ordered = list(targets or [])
        updates = []
        for index in range(feedback_slots):
            if index < len(ordered):
                title = feedback_title(ordered[index])
                updates.extend(
                    [
                        gr.update(visible=True),
                        gr.update(value=f"### {title}\n`{ordered[index]}`", visible=True),
                        gr.update(value=0, visible=True),
                        gr.update(value="", visible=True),
                    ]
                )
            else:
                updates.extend(
                    [
                        gr.update(visible=False),
                        gr.update(value="", visible=False),
                        gr.update(value=0, visible=False),
                        gr.update(value="", visible=False),
                    ]
                )
        return updates

    def run_clicked(
        image_file,
        manual_api_key,
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
        log_queue: queue.Queue[str] = queue.Queue()
        result_box: dict[str, object] = {}

        def emit(message: str) -> None:
            log_queue.put(message)

        def worker() -> None:
            try:
                result_box["value"] = runner.run_session(
                    image_file=image_file,
                    api_key=(manual_api_key or "").strip() or auto_api_key,
                    vlm_model=vlm_model or "gpt-4o",
                    run_v1=run_v1,
                    run_v2=run_v2,
                    run_v3=run_v3,
                    run_hybrid=run_hybrid,
                    vlm_image_only=vlm_image_only,
                    vlm_yolo=vlm_yolo,
                    vlm_sam=vlm_sam,
                    vlm_hybrid=vlm_hybrid,
                    log_callback=emit,
                )
            except Exception as exc:
                result_box["error"] = f"{exc}\n\n{traceback.format_exc()}"

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        logs = ["Calisma basladi. Terminal akisindan sureci izleyebilirsin."]
        yield (
            "",
            [],
            "## Calisiyor\n\nModel ve rapor uretimi devam ediyor.",
            str(runner.db_path),
            "\n".join(logs),
            [],
            *feedback_updates([]),
        )

        while thread.is_alive():
            while True:
                try:
                    logs.append(log_queue.get_nowait())
                except queue.Empty:
                    break
            yield (
                "",
                [],
                "## Calisiyor\n\nModel ve rapor uretimi devam ediyor.",
                str(runner.db_path),
                "\n".join(logs[-250:]),
                [],
                *feedback_updates([]),
            )
            time.sleep(0.5)

        while True:
            try:
                logs.append(log_queue.get_nowait())
            except queue.Empty:
                break

        if "error" in result_box:
            error_text = f"## Hata\n\n```text\n{result_box['error']}\n```"
            logs.append("Hata olustu. Ayrinti yukaridaki traceback icinde.")
            yield (
                "",
                [],
                error_text,
                str(runner.db_path),
                "\n".join(logs[-300:]),
                [],
                *feedback_updates([]),
            )
            return

        session_id, gallery, report, db_path, targets, debug_log = result_box["value"]  # type: ignore[misc]
        logs.append("Calisma tamamlandi.")
        if debug_log:
            logs.extend(str(debug_log).splitlines())
        yield (
            session_id,
            gallery,
            report,
            db_path,
            "\n".join(logs[-300:]),
            targets,
            *feedback_updates(targets),
        )

    def save_all_feedback_clicked(session_id, targets, *values):
        if not session_id:
            raise gr.Error("Once bir test oturumu calistir.")
        saved = []
        target_list = list(targets or [])
        for index, target_name in enumerate(target_list[:feedback_slots]):
            score = values[index * 2]
            comment = (values[index * 2 + 1] or "").strip()
            numeric_score = float(score) if score and float(score) > 0 else None
            stored_comment = comment or ("no feedback" if numeric_score is None else "no comment")
            insert_rating(runner.db_path, session_id, target_name, numeric_score, stored_comment)
            saved.append(f"{target_name}: {numeric_score if numeric_score is not None else 'no feedback'}")
        if not saved:
            return "Kaydedilecek hedef bulunamadi. Once bir analiz calistir."
        return "Kaydedildi:\n" + "\n".join(saved) + f"\n\nDB: {runner.db_path}"

    with gr.Blocks(title="Lejanter Evaluation Interface") as app:
        gr.Markdown("# Lejanter Evaluation Interface")
        gr.Markdown("Resim yukle, modelleri calistir, raporlari gor, puan ve yorumlari Drive SQLite DB'ye kaydet.")
        session_state = gr.State("")
        feedback_targets_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="Test image", type="filepath")
                gr.Markdown(f"**OpenAI API key:** {api_key_status}")
                manual_api_key = gr.Textbox(
                    label="OpenAI API key manuel yedek",
                    type="password",
                    placeholder="Bos birak: Colab Secrets OPENAI_API_KEY kullanilir",
                )
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
                debug_log = gr.Textbox(label="Terminal / Debug log", lines=18, autoscroll=True)

        gr.Markdown("## Puan ve Yorum")
        gr.Markdown("Her ciktinin altinda ayri puan ve yorum var. Puan `0` ise sistem bunu `no feedback` olarak kaydeder.")
        feedback_components = []
        for _ in range(feedback_slots):
            with gr.Group(visible=False) as feedback_group:
                label = gr.Markdown()
                score = gr.Slider(label="Puan (0 = no feedback, 1-5 = degerlendirme)", minimum=0, maximum=5, step=1, value=0)
                comment = gr.Textbox(label="Yorum (opsiyonel)", lines=3)
                feedback_components.extend([feedback_group, label, score, comment])
        save_button = gr.Button("Tum puan/yorumlari kaydet")
        save_status = gr.Textbox(label="Save status")

        run_button.click(
            run_clicked,
            inputs=[
                image,
                manual_api_key,
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
            outputs=[session_state, gallery, report, db_path, debug_log, feedback_targets_state, *feedback_components],
        )
        save_inputs = [session_state, feedback_targets_state]
        for index in range(feedback_slots):
            save_inputs.extend([feedback_components[index * 4 + 2], feedback_components[index * 4 + 3]])
        save_button.click(save_all_feedback_clicked, inputs=save_inputs, outputs=[save_status])
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
    runner = LiveEvaluationRunner(
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
