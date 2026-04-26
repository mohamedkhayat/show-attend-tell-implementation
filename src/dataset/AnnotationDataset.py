#Custom Pytorch Dataset for image cationing tasks.
import json
from pathlib import Path

from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from src.dataset.vocabulary import Vocabulary
from src.utils.data import ensure_flicker_splits, ensure_coco_splits

SPLIT_TYPES = {"train", "val", "test"}


#Class Definition
class AnnotationDataset(Dataset):
    #Constructor
    def __init__(
        self,
        #Path to Dataset folder
        data_path: str,
        split_type="train",
        vocab=None,
        transforms=None,
        max_length=20,
    ):
        if split_type not in SPLIT_TYPES:
            raise ValueError(f"split_type must be one of {sorted(SPLIT_TYPES)}")

        self.data_path = Path(data_path)
        self.split_type = split_type
        self.max_length = max_length
        self.transforms = transforms

        # Load or prepare data splits and captions.
        self.img_paths, self.captions = self._get_or_prepare_data()

        if vocab is None:
            self.vocab = Vocabulary(min_freq=5, max_vocab_size=10000)
            self.vocab.build_vocab(self.captions)
        else:
            self.vocab = vocab

    #Return one training sample
    def __getitem__(self, idx):
        img_path, caption = self.img_paths[idx], self.captions[idx]
        img = np.asarray(Image.open(img_path).convert("RGB"))

        if self.transforms:
            img = self.transforms(img)

        caption_indices = self.vocab.encode(caption, self.max_length)

        return img, caption_indices
    
    #Return number of captions
    def __len__(self):
        return len(self.captions)

    def _get_or_prepare_data(self):
        if "flicker" in str(self.data_path):
            ensure_flicker_splits(str(self.data_path))
        else:
            ensure_coco_splits(str(self.data_path))

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
        """Get up to max_captions reference captions for an image."""
        matching_idxs = [i for i, path in enumerate(self.img_paths) if path == img_path]
        # Remove only exact duplicate caption strings while keeping original order.
        unique_captions = list(dict.fromkeys(self.captions[i] for i in matching_idxs))
        return unique_captions[:max_captions]
