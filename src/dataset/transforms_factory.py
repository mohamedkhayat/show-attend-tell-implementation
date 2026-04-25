import torchvision.transforms.v2 as v2
from src.dataset.model_factory import MODEL_WEIGHTS_MAPPING


def get_transforms(model):
    weights = MODEL_WEIGHTS_MAPPING[model].DEFAULT
    model_preprocess = weights.transforms()
    train_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            v2.RandomGrayscale(p=0.05),
            model_preprocess,
        ]
    )
    test_transforms = v2.Compose([v2.ToImage(), model_preprocess])
    return train_transforms, test_transforms
