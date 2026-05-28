from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def bbox_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = bbox_area([ix1, iy1, ix2, iy2])
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def parse_thresholds(values: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected class=threshold, got {value}")
        label, threshold = value.split("=", 1)
        thresholds[label.strip()] = float(threshold)
    return thresholds


def class_threshold(label: str, default: float, thresholds: dict[str, float]) -> float:
    return thresholds.get(label, default)


def nms_by_class(detections: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        by_class[detection["label"]].append(detection)

    kept: list[dict[str, Any]] = []
    for label_detections in by_class.values():
        ordered = sorted(label_detections, key=lambda item: item.get("confidence", 0.0), reverse=True)
        selected: list[dict[str, Any]] = []
        for candidate in ordered:
            if all(bbox_iou(candidate["bbox_xyxy"], chosen["bbox_xyxy"]) < iou_threshold for chosen in selected):
                selected.append(candidate)
        kept.extend(selected)
    return sorted(kept, key=lambda item: item.get("confidence", 0.0), reverse=True)


def filter_record(
    record: dict[str, Any],
    default_conf: float,
    class_thresholds: dict[str, float],
    min_area_ratio: float,
    max_area_ratio: float,
    nms_iou: float,
) -> dict[str, Any]:
    image_area = float(record["width"] * record["height"])
    candidates = []
    removed = defaultdict(int)

    for detection in record.get("detections", []):
        label = detection["label"]
        confidence = float(detection.get("confidence", 0.0))
        threshold = class_threshold(label, default_conf, class_thresholds)
        area_ratio = bbox_area(detection["bbox_xyxy"]) / image_area if image_area > 0 else 0.0

        if confidence < threshold:
            removed["low_confidence"] += 1
            continue
        if area_ratio < min_area_ratio:
            removed["too_small"] += 1
            continue
        if area_ratio > max_area_ratio:
            removed["too_large"] += 1
            continue

        kept_detection = dict(detection)
        kept_detection["filter_conf_threshold"] = threshold
        kept_detection["bbox_area_ratio"] = round(area_ratio, 6)
        candidates.append(kept_detection)

    before_nms = len(candidates)
    filtered = nms_by_class(candidates, nms_iou)
    removed["duplicate_nms"] += before_nms - len(filtered)

    output = dict(record)
    output["detections"] = filtered
    output["filter_summary"] = {
        "raw_count": len(record.get("detections", [])),
        "kept_count": len(filtered),
        "removed": dict(removed),
        "default_conf": default_conf,
        "class_thresholds": class_thresholds,
        "min_area_ratio": min_area_ratio,
        "max_area_ratio": max_area_ratio,
        "nms_iou": nms_iou,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter YOLO detections before SAM2 refinement.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--default-conf", type=float, default=0.7)
    parser.add_argument("--class-threshold", action="append", default=[], help="Per-class threshold like cam=0.75")
    parser.add_argument("--min-area-ratio", type=float, default=0.00005)
    parser.add_argument("--max-area-ratio", type=float, default=0.35)
    parser.add_argument("--nms-iou", type=float, default=0.35)
    args = parser.parse_args()

    class_thresholds = parse_thresholds(args.class_threshold)
    records = json.loads(args.input.read_text(encoding="utf-8"))
    filtered = [
        filter_record(
            record,
            default_conf=args.default_conf,
            class_thresholds=class_thresholds,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
            nms_iou=args.nms_iou,
        )
        for record in records
    ]

    summary = {
        "images": len(filtered),
        "raw_detections": sum(item["filter_summary"]["raw_count"] for item in filtered),
        "kept_detections": sum(item["filter_summary"]["kept_count"] for item in filtered),
        "removed": {},
    }
    removed_totals: defaultdict[str, int] = defaultdict(int)
    for item in filtered:
        for reason, count in item["filter_summary"]["removed"].items():
            removed_totals[reason] += count
    summary["removed"] = dict(removed_totals)

    payload = {
        "summary": summary,
        "records": filtered,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
