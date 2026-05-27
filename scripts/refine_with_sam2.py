from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


def mask_to_polygons(mask: np.ndarray) -> list[list[list[float]]]:
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[float]]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = [[round(float(x), 3), round(float(y), 3)] for [[x, y]] in approx]
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = [
        (255, 80, 80),
        (80, 180, 255),
        (80, 220, 120),
        (255, 200, 80),
        (200, 100, 255),
        (255, 120, 200),
    ]
    return palette[index % len(palette)]


def draw_overlay(image_rgb: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    overlay = image_rgb.copy()
    canvas = image_rgb.copy()
    for index, detection in enumerate(detections):
        mask = np.array(detection.get("_sam2_mask", []), dtype=bool)
        if mask.size == 0:
            continue
        color = np.array(color_for_index(index), dtype=np.uint8)
        canvas[mask] = (0.55 * canvas[mask] + 0.45 * color).astype(np.uint8)
        x1, y1, x2, y2 = [int(round(v)) for v in detection["bbox_xyxy"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_for_index(index), 2)
        label = f"{detection['label']} {detection['confidence']:.2f}"
        cv2.putText(
            canvas,
            label,
            (max(0, x1), max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color_for_index(index),
            2,
            cv2.LINE_AA,
        )
    overlay[:] = canvas
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine YOLO boxes with SAM2 masks.")
    parser.add_argument("--detections", required=True, type=Path, help="YOLO detection JSON from infer_yolo.py.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="SAM2 checkpoint path.")
    parser.add_argument("--model-cfg", required=True, help="SAM2 config, e.g. configs/sam2.1/sam2.1_hiera_t.yaml.")
    parser.add_argument("--out", required=True, type=Path, help="Output JSON with SAM2 polygons.")
    parser.add_argument("--visual-dir", required=True, type=Path, help="Folder for SAM2 overlay images.")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    args = parser.parse_args()

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    model = build_sam2(args.model_cfg, str(args.checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)

    records = json.loads(args.detections.read_text(encoding="utf-8"))
    refined_records: list[dict[str, Any]] = []
    args.visual_dir.mkdir(parents=True, exist_ok=True)

    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    autocast_enabled = device.startswith("cuda")

    for record in records:
        image_path = Path(record["image"])
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        predictor.set_image(image_rgb)

        refined_detections: list[dict[str, Any]] = []
        with torch.inference_mode(), torch.autocast(autocast_device, dtype=torch.bfloat16, enabled=autocast_enabled):
            for detection in record.get("detections", []):
                box = np.array(detection["bbox_xyxy"], dtype=np.float32)
                masks, scores, _ = predictor.predict(box=box, multimask_output=False)
                mask = masks[0].astype(bool)
                refined = dict(detection)
                refined["sam2_score"] = round(float(scores[0]), 4)
                refined["sam2_mask_area_px"] = int(mask.sum())
                refined["sam2_polygons"] = mask_to_polygons(mask)
                refined["_sam2_mask"] = mask
                refined_detections.append(refined)

        overlay = draw_overlay(image_rgb, refined_detections)
        visual_path = args.visual_dir / f"sam2_refined__{image_path.name}"
        Image.fromarray(overlay).save(visual_path)

        serializable_detections = []
        for detection in refined_detections:
            clean = dict(detection)
            clean.pop("_sam2_mask", None)
            serializable_detections.append(clean)

        refined_records.append(
            {
                "image": str(image_path),
                "width": int(record["width"]),
                "height": int(record["height"]),
                "source_detections": str(args.detections),
                "sam2_checkpoint": str(args.checkpoint),
                "visual": str(visual_path),
                "detections": serializable_detections,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(refined_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote visuals to {args.visual_dir}")


if __name__ == "__main__":
    main()
