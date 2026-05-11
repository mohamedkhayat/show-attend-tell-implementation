from torchvision.models import (
    VGG19_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)

# Maps encoder name → torchvision Weights class (used by transforms_factory)
MODEL_WEIGHTS_MAPPING = {
    "vgg19":    VGG19_Weights,
    "resnet50": ResNet50_Weights,
    "resnet101": ResNet101_Weights,
}
