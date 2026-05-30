from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from evaluation_interface import EvaluationRunner, VLM_MODELS, insert_rating


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


def build_app(runner: EvaluationRunner):
    import gradio as gr

    auto_api_key, api_key_status = get_openai_api_key()

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
        try:
            session_id, gallery, report, db_path, targets, debug_log = runner.run_session(
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
            outputs=[session_state, gallery, report, db_path, debug_log, target],
        )
        save_button.click(save_rating_clicked, inputs=[session_state, target, score, comment], outputs=[save_status])
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
