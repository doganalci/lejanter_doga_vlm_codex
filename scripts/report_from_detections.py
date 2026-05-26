from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def image_key(path: str) -> str:
    return Path(path).name


def assign_to_facades(
    element_records: list[dict[str, Any]], facade_records: list[dict[str, Any]]
) -> dict[str, dict[str, Counter]]:
    facades_by_image = {image_key(record["image"]): record for record in facade_records}
    counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    for element_record in element_records:
        key = image_key(element_record["image"])
        facade_record = facades_by_image.get(key)
        facade_detections = facade_record.get("detections", []) if facade_record else []

        for detection in element_record.get("detections", []):
            label = detection["label"]
            center = bbox_center(detection["bbox_xyxy"])
            assigned = "unassigned"
            for index, facade in enumerate(facade_detections, start=1):
                polygon = facade.get("polygon", [])
                if polygon and point_in_polygon(center, polygon):
                    assigned = f"facade_{index}"
                    break
            counts[key][assigned][label] += 1

    return counts


def render_markdown(counts: dict[str, dict[str, Counter]]) -> str:
    lines = ["# Mimari Cephe Lejant Raporu", ""]
    if not counts:
        lines.append("Tespit sonucu bulunamadı.")
        return "\n".join(lines) + "\n"

    for image_name, facade_counts in sorted(counts.items()):
        lines.extend([f"## {image_name}", ""])
        for facade_id, counter in sorted(facade_counts.items()):
            total = sum(counter.values())
            lines.append(f"### {facade_id}")
            lines.append("")
            lines.append(f"Toplam {total} cephe elemanı tespit edildi.")
            lines.append("")
            for label, count in counter.most_common():
                lines.append(f"- {label}: {count}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown facade report from detection JSON files.")
    parser.add_argument("--elements", required=True, type=Path)
    parser.add_argument("--facades", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    elements = load_json(args.elements)
    facades = load_json(args.facades)
    counts = assign_to_facades(elements, facades)
    report = render_markdown(counts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
