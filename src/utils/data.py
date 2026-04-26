# =============================================================================
# data.py — Préparation et gestion des splits de données
#
# Rôle : parser les fichiers bruts de Flickr8k et générer des fichiers JSON
# reproductibles pour les splits train/val/test.
#
# Structure attendue du dossier Flickr8k :
#   data/flicker8k/
#   ├── captions.txt     ← colonnes "image" et "caption" séparées par virgule
#   └── Images/          ← 8091 fichiers .jpg
#
# FIX appliqué :
#   ERREUR rencontrée au premier lancement :
#   "FileNotFoundError: Could not find captions file at data/flicker8k/captions.txt"
#   CAUSE : mauvais dataset Kaggle téléchargé (certains ont "captions.csv" ou
#   une structure différente). Le bon dataset est "adityajn105/flickr8k" qui
#   contient exactement captions.txt + Images/ avec les colonnes "image","caption".
#
# Division appliquée (seed=42 pour reproductibilité) :
#   Train : 80% → 6472 images → 32 360 paires (image, légende) [5 par image]
#   Val   : 10% →  809 images →  4 045 paires
#   Test  : 10% →  809 images →  4 045 paires
#
# Idempotence : ensure_flicker_splits() ne régénère les splits que si les
# fichiers JSON sont absents — évite de re-parser à chaque import.
# =============================================================================

import json
from pathlib import Path
from random import Random

import pandas as pd

SPLIT_TYPES = ("train", "val", "test")


def _split_images(
    images: list[str], split_ratios: tuple[float, float, float]
) -> tuple[list[str], list[str], list[str]]:
    """Divise la liste d'images en trois groupes selon les ratios donnés."""
    train_ratio, val_ratio, _ = split_ratios
    total = len(images)

    train_end = int(total * train_ratio)
    val_end   = train_end + int(total * val_ratio)

    train_images = images[:train_end]
    val_images   = images[train_end:val_end]
    test_images  = images[val_end:]
    return train_images, val_images, test_images


def prepare_flicker_data(
    base_path: str,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> None:
    """Parse captions.txt et génère les fichiers JSON de splits.

    Pour chaque split, produit deux fichiers JSON parallèles :
      - <split>_img_paths.json : liste des chemins absolus vers les images
      - <split>_captions.json  : liste des légendes correspondantes

    Les deux listes ont la même longueur et sont alignées index par index :
      img_paths[i] ↔ captions[i]

    Chaque image apparaît 5 fois (une fois par légende humaine).
    """
    base_dir      = Path(base_path)
    captions_path = base_dir / "captions.txt"
    images_dir    = base_dir / "Images"

    if not captions_path.exists():
        raise FileNotFoundError(f"Could not find captions file at {captions_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Could not find images directory at {images_dir}")

    if abs(sum(split_ratios) - 1.0) > 1e-8:
        raise ValueError("split_ratios must sum to 1.0")

    captions_df = pd.read_csv(captions_path)

    # Vérifie que le CSV a bien les colonnes attendues (fix pour mauvais datasets Kaggle)
    required_cols = {"image", "caption"}
    if not required_cols.issubset(captions_df.columns):
        raise ValueError("captions.txt must contain columns 'image' and 'caption'")

    captions_df = captions_df.dropna(subset=["image", "caption"])

    # Regroupe les 5 légendes par image dans un dictionnaire
    img_to_captions: dict[str, list[str]] = {}
    for row in captions_df.itertuples(index=False):
        image_name = str(row.image).strip()
        caption    = str(row.caption).strip()
        if not image_name or not caption:
            continue
        img_to_captions.setdefault(image_name, []).append(caption)

    # sorted() garantit un ordre déterministe avant le shuffle
    all_images = sorted(img_to_captions.keys())
    if not all_images:
        raise ValueError("No image-caption pairs found in captions.txt")

    # seed=42 → même shuffle à chaque exécution → splits reproductibles
    rng = Random(seed)
    rng.shuffle(all_images)

    train_images, val_images, test_images = _split_images(all_images, split_ratios)

    for split_name, split_images in (
        ("train", train_images),
        ("val",   val_images),
        ("test",  test_images),
    ):
        img_paths: list[str] = []
        captions:  list[str] = []

        for image_name in split_images:
            full_image_path = images_dir / image_name
            if not full_image_path.exists():
                continue  # ignore les images référencées mais absentes du dossier

            # Chaque image est répétée autant de fois qu'elle a de légendes (5 en général)
            for caption in img_to_captions[image_name]:
                img_paths.append(str(full_image_path))
                captions.append(caption)

        with (base_dir / f"{split_name}_img_paths.json").open("w") as f:
            json.dump(img_paths, f)

        with (base_dir / f"{split_name}_captions.json").open("w") as f:
            json.dump(captions, f)


def ensure_flicker_splits(base_path: str) -> None:
    """Génère les splits uniquement si les fichiers JSON sont absents.

    Appelé automatiquement par AnnotationDataset à chaque instanciation.
    Si les 6 fichiers JSON existent déjà, ne fait rien (idempotent).
    """
    base_dir = Path(base_path)
    required = (
        [base_dir / f"{s}_img_paths.json" for s in SPLIT_TYPES] +
        [base_dir / f"{s}_captions.json"  for s in SPLIT_TYPES]
    )

    if all(path.exists() for path in required):
        return  # déjà générés → on ne retouche pas

    prepare_flicker_data(base_path)


def prepare_coco_data(base_path: str) -> None:
    # Support MSCOCO non implémenté — prévu pour une version future
    raise NotImplementedError(
        f"MSCOCO preparation is not implemented yet. Received base_path={base_path!r}."
    )


def ensure_coco_splits(base_path: str) -> None:
    # Support MSCOCO non implémenté — prévu pour une version future
    raise NotImplementedError(
        f"MSCOCO preparation is not implemented yet. Received base_path={base_path!r}."
    )
