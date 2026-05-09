import logging

import torch
import torch.nn as nn

from src.utils.logging import configure_logging
import torchvision.models as models

logger = logging.getLogger(__name__)
backbones = {"vgg19": models.vgg19(weights=models.VGG19_Weights.DEFAULT)}


class Encoder(nn.Module):
    def __init__(self, device, model_name):
        super().__init__()
        backbone = backbones[model_name].to(device)
        self.features = backbone.features[:29]

    def forward(self, x):
        x = self.features(x)  # (B, 3, 224, 224)
        x = x.permute(0, 2, 3, 1)  # (B, 14, 14, 512)
        x = x.view(x.size(0), -1, x.size(-1))  # (B, 196, 512)
        logger.info("Encoder output shape | %s", tuple(x.shape))
        return x


if __name__ == "__main__":
    configure_logging("DEBUG")
    model = Encoder({"model": "vgg19", "weights": "VGG19_Weights.DEFAULT"})
    inp = torch.zeros(32, 3, 224, 224)
    model(inp)
