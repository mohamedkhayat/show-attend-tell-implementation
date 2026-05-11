import torch
import torch.nn as nn

from src.models.decoder import Decoder
from src.models.encoder import Encoder


class Model(nn.Module):
    def __init__(
        self,
        device,
        enc_model_name: str,
        max_seq_len: int = 40,
        dropout_prob: float = 0.5,
        use_tf: bool = True,
    ):
        super().__init__()
        self.enc = Encoder(enc_model_name)
        # Decoder reads encoder's output dimension automatically
        self.dec = Decoder(
            device,
            encoder_dim=self.enc.output_dim,
            max_seq_len=max_seq_len,
            dropout_prob=dropout_prob,
            use_tf=use_tf,
        )

    def forward(self, images: torch.Tensor, captions=None):
        # Skip gradient computation for the frozen encoder to save activation memory
        enc_params_frozen = not any(p.requires_grad for p in self.enc.parameters())
        if enc_params_frozen:
            with torch.no_grad():
                annot_vecs = self.enc(images)
        else:
            annot_vecs = self.enc(images)

        preds, alphas = self.dec(annot_vecs, captions)
        return preds, alphas
