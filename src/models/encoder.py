import logging

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = torch.hub.load(
            "pytorch/vision", cfg["model"], weights=cfg["weights"]
        )
        self.features = backbone.features[:28]

    def forward(self, x):
        x = self.features(x)  # (B, 512, 14, 14)
        x = x.permute(0, 2, 3, 1)  # (B, 14, 14, 512)
        x = x.view(x.size(0), -1, x.size(-1))  # (B, 196, 512)
        logger.debug("Encoder output shape | %s", tuple(x.shape))
