import logging

import numpy as np
import torch

from src.dataset.AnnotationDataset import AnnotationDataset
from src.models.encoder import Encoder
from src.models.model import Model
from src.utils.logging import configure_logging
from src.models.decoder import Decoder

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    configure_logging("DEBUG")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp = torch.zeros(32, 3, 224, 224).to(device)
    captions = torch.zeros(32, 124).long().to(device)
    model = Model(device, "vgg19").to(device)
    output = model(inp, captions)
