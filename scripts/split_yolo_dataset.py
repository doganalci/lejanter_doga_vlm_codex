from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def image_files(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def ensure_split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, label_dir


def paired_label(image: Path, label_dir: Path) -> Path:
    return label_dir / f"{image.stem}.txt"


def copy_pair(image: Path, source_label_dir: Path, target_image_dir: Path, target_label_dir: Path) -> None:
    label = paired_label(image, source_label_dir)
    shutil.copy2(image, target_image_dir / image.name)
    if label.exists():
        shutil.copy2(label, target_label_dir / label.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/valid/test folders from a Roboflow YOLO train-only export.")
    parser.add_argument("--root", required=True, type=Path, help="Dataset root containing train/images and train/labels.")
    parser.add_argument("--valid", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument("--test", type=float, default=0.1, help="Test fraction.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy instead of moving files out of train.")
    args = parser.parse_args()

    root = args.root
    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    if not train_images.exists() or not train_labels.exists():
        raise SystemExit(f"Missing train/images or train/labels under {root}")

    images = image_files(train_images)
    if not images:
        raise SystemExit(f"No images found in {train_images}")

    random.Random(args.seed).shuffle(images)
    test_count = max(1, round(len(images) * args.test))
    valid_count = max(1, round(len(images) * args.valid))
    test_images = images[:test_count]
    valid_images = images[test_count : test_count + valid_count]

    for split, split_images in (("valid", valid_images), ("test", test_images)):
        target_images, target_labels = ensure_split_dirs(root, split)
        for image in split_images:
            copy_pair(image, train_labels, target_images, target_labels)
            if not args.copy:
                label = paired_label(image, train_labels)
                image.unlink()
                if label.exists():
                    label.unlink()

    print(f"Total: {len(images)}")
    print(f"Train: {len(image_files(train_images))}")
    print(f"Valid: {len(image_files(root / 'valid' / 'images'))}")
    print(f"Test: {len(image_files(root / 'test' / 'images'))}")


if __name__ == "__main__":
    main()
