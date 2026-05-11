import torch
import torchvision.transforms.v2 as v2

from src.models.model_factory import MODEL_WEIGHTS_MAPPING

# CLIP-specific normalisation (differs from ImageNet)
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


def _clip_base_transform():
    return v2.Compose([
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ])


def get_transforms(model_name: str):
    """Return (train_transforms, test_transforms) for the given encoder name."""
    if model_name.startswith("clip"):
        base = _clip_base_transform()
        train_transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            v2.RandomGrayscale(p=0.05),
            base,
        ])
        return train_transforms, base

    # VGG19 / ResNet — use the weights' built-in transform (ImageNet stats)
    weights = MODEL_WEIGHTS_MAPPING[model_name].DEFAULT
    base = weights.transforms()
    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        v2.RandomGrayscale(p=0.05),
        base,
    ])
    return train_transforms, base
