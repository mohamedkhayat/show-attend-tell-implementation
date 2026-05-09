import torch
import torch.nn as nn

from src.models.decoder import Decoder
from src.models.encoder import Encoder


class Model(nn.Module):
    def __init__(
        self,
        device,
        enc_model_name,
        encoder_dim=512,
        max_seq_len=124,
        dropout_prob=0.3,
        use_tf=False,
    ):
        super().__init__()
        self.enc = Encoder(device, enc_model_name)
        self.dec = Decoder(device, encoder_dim, max_seq_len, dropout_prob, use_tf)

    def forward(self, images, captions):
        annot_vecs = self.enc(images)
        preds = self.dec(annot_vecs, captions)
        return preds
