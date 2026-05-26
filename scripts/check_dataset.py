from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def count_files(path: Path, suffixes: tuple[str, ...]) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in suffixes)


def resolve_split(dataset_root: Path, split_value: str) -> Path:
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path
    return dataset_root / split_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a YOLO dataset yaml exported from Roboflow.")
    parser.add_argument("--data", required=True, type=Path, help="Path to dataset yaml.")
    args = parser.parse_args()

    data = read_yaml(args.data)
    dataset_root = Path(data.get("path", ".")).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (args.data.parent / dataset_root).resolve()

    print(f"Dataset yaml: {args.data}")
    print(f"Dataset root: {dataset_root}")
    print(f"Classes: {data.get('names', {})}")

    for split in ("train", "val", "valid", "test"):
        if split not in data:
            continue
        image_dir = resolve_split(dataset_root, data[split])
        label_dir = Path(str(image_dir).replace("/images", "/labels"))
        images = count_files(image_dir, (".jpg", ".jpeg", ".png", ".webp"))
        labels = count_files(label_dir, (".txt",))
        print(f"{split}: {images} images, {labels} labels")


if __name__ == "__main__":
    main()
