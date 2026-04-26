# =============================================================================
# AnnotationDataset.py — Dataset PyTorch pour Flickr8k / MSCOCO
#
# Rôle : fournir des paires (image_tensor, caption_indices) compatibles avec
# torch.utils.data.DataLoader pour l'entraînement et la validation.
#
# Fonctionnement :
#   1. Au premier appel, génère les fichiers split JSON (train/val/test)
#      à partir de captions.txt si ils n'existent pas encore.
#   2. Construit automatiquement le vocabulaire depuis les légendes train
#      (ou réutilise un vocab existant passé en paramètre pour val/test).
#   3. __getitem__ charge l'image, applique les transforms, encode la légende.
#
# FIX appliqué sur les transforms :
#   Les images sont chargées via PIL puis converties en numpy array (np.asarray).
#   Le fix v2.ToImage() dans transforms_factory.py permet de convertir ce numpy
#   array en tensor torchvision avant les autres transforms.
#   Sans ce fix : "TypeError: Unexpected type <class 'numpy.ndarray'>"
#
# Note importante sur le vocab partagé :
#   Pour val et test, on DOIT passer vocab=train_ds.vocab — le modèle ne peut
#   pas voir de mots inconnus du vocabulaire d'entraînement.
#   Exemple dans train.py :
#     train_ds = AnnotationDataset(..., split_type="train")
#     val_ds   = AnnotationDataset(..., split_type="val", vocab=train_ds.vocab)
# =============================================================================

import json
from pathlib import Path

from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from src.dataset.vocabulary import Vocabulary
from src.utils.data import ensure_flicker_splits, ensure_coco_splits

SPLIT_TYPES = {"train", "val", "test"}


class AnnotationDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split_type="train",
        vocab=None,
        transforms=None,
        max_length=20,
    ):
        """
        Args:
            data_path   : chemin vers le dossier du dataset (ex: "data/flicker8k")
            split_type  : "train", "val" ou "test"
            vocab       : vocabulaire pré-construit (obligatoire pour val/test)
                          si None, un nouveau vocabulaire est construit depuis les légendes
            transforms  : pipeline de transforms à appliquer sur les images
            max_length  : longueur max des séquences tokenisées (avec padding)
        """
        if split_type not in SPLIT_TYPES:
            raise ValueError(f"split_type must be one of {sorted(SPLIT_TYPES)}")

        self.data_path  = Path(data_path)
        self.split_type = split_type
        self.max_length = max_length
        self.transforms = transforms

        # Génère les splits JSON si manquants, puis les charge
        self.img_paths, self.captions = self._get_or_prepare_data()

        if vocab is None:
            # Construction du vocabulaire uniquement depuis les données d'entraînement
            self.vocab = Vocabulary(min_freq=5, max_vocab_size=10000)
            self.vocab.build_vocab(self.captions)
        else:
            # Réutilise le vocab du train — indispensable pour val/test
            self.vocab = vocab

    def __getitem__(self, idx):
        """Retourne une paire (image_tensor, caption_indices) pour l'index donné.

        L'image est chargée en RGB puis convertie en numpy array.
        Le numpy array est ensuite transformé par les transforms (qui incluent
        v2.ToImage() pour la conversion numpy → tensor).
        """
        img_path, caption = self.img_paths[idx], self.captions[idx]

        # np.asarray(PIL.Image) → numpy array (H, W, 3) en uint8
        # Note : retourne un tableau non-modifiable (read-only) d'où le UserWarning
        # "NumPy array is not writable" — inoffensif, supprimé après le premier avertissement
        img = np.asarray(Image.open(img_path).convert("RGB"))

        if self.transforms:
            img = self.transforms(img)  # numpy → tensor via v2.ToImage() dans les transforms

        # Encode la légende : "a dog" → [1, 4, 87, 2, 0, 0, ...] (avec <start>, <end>, <pad>)
        caption_indices = self.vocab.encode(caption, self.max_length)

        return img, caption_indices

    def __len__(self):
        return len(self.captions)

    def _get_or_prepare_data(self):
        """Génère les fichiers split si manquants, puis les charge."""
        # Détecte le dataset depuis le nom du dossier
        if "flicker" in str(self.data_path):
            ensure_flicker_splits(str(self.data_path))
        else:
            ensure_coco_splits(str(self.data_path))  # lève NotImplementedError pour COCO

        img_path_file = self.data_path / f"{self.split_type}_img_paths.json"
        captions_file = self.data_path / f"{self.split_type}_captions.json"

        with img_path_file.open("r") as f:
            img_paths = json.load(f)

        with captions_file.open("r") as f:
            captions = json.load(f)

        if len(img_paths) != len(captions):
            raise ValueError(
                "Mismatch between number of image paths and captions in split files"
            )

        return img_paths, captions

    def get_all_captions_for_image(self, img_path, max_captions=5):
        """Retourne jusqu'à 5 légendes de référence pour une image donnée.

        Utilisé dans evaluate.py pour calculer le score BLEU contre toutes
        les références humaines disponibles (Flickr8k en a 5 par image).
        Les doublons exacts sont supprimés tout en préservant l'ordre.
        """
        matching_idxs = [i for i, path in enumerate(self.img_paths) if path == img_path]
        unique_captions = list(dict.fromkeys(self.captions[i] for i in matching_idxs))
        return unique_captions[:max_captions]
