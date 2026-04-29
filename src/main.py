import logging

import numpy as np
import torch

from src.dataset.AnnotationDataset import AnnotationDataset
from src.utils.logging import configure_logging
from src.models.decoder import Decoder

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    configure_logging("DEBUG")

    # from torchvision.models import VGG19_Weights, vgg19

    # logger.info("Loading VGG19 backbone with default pretrained weights")
    # model = vgg19(weights=VGG19_Weights.DEFAULT)

    # logger.debug("Model summary:\n%s", model)
    # from src.models.attention import Attention

    # att = Attention(encoder_dim=512)
    # prev_hidden = torch.randn(32, 512)
    # rt = torch.randn(32, 196, 512)
    # # att(rt, prev_hidden)
    # ds = AnnotationDataset("data/flicker8k")
