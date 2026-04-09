from __future__ import annotations

import json
from pathlib import Path
from random import Random

import pandas as pd

SPLIT_TYPES = ("train", "val", "test")


def _split_images(
    images: list[str], split_ratios: tuple[float, float, float]
) -> tuple[list[str], list[str], list[str]]:
    train_ratio, val_ratio, _ = split_ratios
    total = len(images)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    return train_images, val_images, test_images


def prepare_flicker_data(
    base_path: str,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> None:
    """Build split JSON files from Flicker8k captions.txt.

    The output files are:
    - <split>_img_paths.json
    - <split>_captions.json
    """
    base_dir = Path(base_path)
    captions_path = base_dir / "captions.txt"
    images_dir = base_dir / "Images"

    if not captions_path.exists():
        raise FileNotFoundError(f"Could not find captions file at {captions_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Could not find images directory at {images_dir}")

    if abs(sum(split_ratios) - 1.0) > 1e-8:
        raise ValueError("split_ratios must sum to 1.0")

    captions_df = pd.read_csv(captions_path)
    required_cols = {"image", "caption"}
    if not required_cols.issubset(captions_df.columns):
        raise ValueError("captions.txt must contain columns 'image' and 'caption'")

    captions_df = captions_df.dropna(subset=["image", "caption"])

    img_to_captions: dict[str, list[str]] = {}
    for row in captions_df.itertuples(index=False):
        image_name = str(row.image).strip()
        caption = str(row.caption).strip()
        if not image_name or not caption:
            continue
        img_to_captions.setdefault(image_name, []).append(caption)

    all_images = sorted(img_to_captions.keys())
    if not all_images:
        raise ValueError("No image-caption pairs found in captions.txt")

    rng = Random(seed)
    rng.shuffle(all_images)

    train_images, val_images, test_images = _split_images(all_images, split_ratios)

    for split_name, split_images in (
        ("train", train_images),
        ("val", val_images),
        ("test", test_images),
    ):
        img_paths: list[str] = []
        captions: list[str] = []

        for image_name in split_images:
            full_image_path = images_dir / image_name
            if not full_image_path.exists():
                continue
            for caption in img_to_captions[image_name]:
                img_paths.append(str(full_image_path))
                captions.append(caption)

        with (base_dir / f"{split_name}_img_paths.json").open("w") as f:
            json.dump(img_paths, f)

        with (base_dir / f"{split_name}_captions.json").open("w") as f:
            json.dump(captions, f)


def ensure_flicker_splits(base_path: str) -> None:
    """Generate split files only if they are missing."""
    base_dir = Path(base_path)
    required = [
        base_dir / f"{split_type}_img_paths.json" for split_type in SPLIT_TYPES
    ] + [base_dir / f"{split_type}_captions.json" for split_type in SPLIT_TYPES]

    if all(path.exists() for path in required):
        return

    prepare_flicker_data(base_path)


def prepare_coco_data(base_path: str) -> None:
    raise NotImplementedError(
        f"MSCOCO preparation is not implemented yet. Received base_path={base_path!r}."
    )


def ensure_coco_splits(base_path: str) -> None:
    """Generate split files only if they are missing."""
    raise NotImplementedError(
        f"MSCOCO preparation is not implemented yet. Received base_path={base_path!r}."
    )
