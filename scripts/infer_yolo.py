from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def xyxy_to_list(values: Any) -> list[float]:
    return [round(float(item), 3) for item in values]


def mask_to_polygon(mask: Any) -> list[list[float]]:
    if mask is None:
        return []
    return [[round(float(x), 3), round(float(y), 3)] for x, y in mask]


def result_to_record(result: Any) -> dict[str, Any]:
    names = result.names
    image_record: dict[str, Any] = {
        "image": str(result.path),
        "width": int(result.orig_shape[1]),
        "height": int(result.orig_shape[0]),
        "detections": [],
    }

    boxes = result.boxes
    polygons = []
    if result.masks is not None and result.masks.xy is not None:
        polygons = result.masks.xy

    if boxes is None:
        return image_record

    for index, box in enumerate(boxes):
        class_id = int(box.cls.item())
        detection = {
            "label": names.get(class_id, str(class_id)),
            "class_id": class_id,
            "confidence": round(float(box.conf.item()), 4),
            "bbox_xyxy": xyxy_to_list(box.xyxy[0].tolist()),
        }
        if index < len(polygons):
            detection["polygon"] = mask_to_polygon(polygons[index])
        image_record["detections"].append(detection)

    return image_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference and save normalized JSON.")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--task", choices=("detect", "segment"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default=None, help="Run name under outputs/runs.")
    parser.add_argument("--project", type=Path, default=Path("outputs/runs"), help="Output project directory.")
    parser.add_argument("--save-visuals", action="store_true")
    args = parser.parse_args()

    model = YOLO(str(args.weights), task=args.task)
    results = model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name or f"infer-{args.weights.stem}",
        save=args.save_visuals,
        exist_ok=True,
    )

    payload = [result_to_record(result) for result in results]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
