import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights


class Encoder(nn.Module):
    """CNN encoder based on VGG19.

    Extracts spatial feature maps from the 4th conv block (before the last
    max-pool), giving a 14x14 grid of 512-dim annotation vectors — exactly
    as described in the paper (Sec. 3.1.1 & 4.3).

    Output shape: (batch, L, D) where L=196 (14x14) and D=512.
    """

    def __init__(self, encoded_dim: int = 512, fine_tune: bool = False):
        super().__init__()

        vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        # Keep only up to conv4_3 (features[0..27], before the 4th max-pool)
        # VGG19 features layout:
        #   [0..4]   → block1 (conv+relu x2 + maxpool)
        #   [5..9]   → block2 (conv+relu x2 + maxpool)
        #   [10..18] → block3 (conv+relu x4 + maxpool)
        #   [19..27] → block4 (conv+relu x4) ← we stop here (before maxpool[28])
        self.features = nn.Sequential(*list(vgg19.features.children())[:29])

        # Freeze all encoder weights by default (paper uses frozen VGG19)
        self.set_fine_tune(fine_tune)

        self.encoded_dim = encoded_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch, 3, 224, 224) — normalized ImageNet tensors

        Returns:
            features: (batch, 196, 512) — L=14*14 annotation vectors
        """
        out = self.features(images)          # (B, 512, 14, 14)
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1)        # (B, 14, 14, 512)
        out = out.reshape(B, H * W, C)       # (B, 196, 512)
        return out

    def set_fine_tune(self, fine_tune: bool) -> None:
        """Allow or freeze gradient updates for the CNN weights."""
        for param in self.features.parameters():
            param.requires_grad = fine_tune
