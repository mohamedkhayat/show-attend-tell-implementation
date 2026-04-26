# =============================================================================
# encoder.py — Encodeur CNN basé sur VGG19
#
# Rôle : transformer une image brute en une grille de vecteurs visuels (annotation
# vectors) que le décodeur LSTM pourra "regarder" à chaque étape de génération.
#
# Principe du papier (Sec. 3.1.1 & 4.3) :
#   On utilise VGG19 pré-entraîné et on s'arrête après le 4ème bloc convolutif,
#   AVANT le dernier max-pooling. Cela préserve une résolution spatiale 14×14
#   au lieu de l'écraser à 7×7. Chaque position (i,j) de cette grille correspond
#   à une région de l'image → 196 "annotation vectors" de dimension 512.
# =============================================================================

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights


class Encoder(nn.Module):
    """CNN encoder based on VGG19.

    Input  : (B, 3, 224, 224) — batch d'images normalisées ImageNet
    Output : (B, 196, 512)    — 196 annotation vectors (grille 14×14 × 512 dims)
    """

    def __init__(self, encoded_dim: int = 512, fine_tune: bool = False):
        super().__init__()

        vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        # On garde uniquement les blocs conv 1 à 4 de VGG19 (indices 0..28).
        # Structure de vgg19.features :
        #   [0..4]   → bloc 1 : 2×(Conv+ReLU) + MaxPool  → sortie 112×112
        #   [5..9]   → bloc 2 : 2×(Conv+ReLU) + MaxPool  → sortie 56×56
        #   [10..18] → bloc 3 : 4×(Conv+ReLU) + MaxPool  → sortie 28×28
        #   [19..27] → bloc 4 : 4×(Conv+ReLU)            → sortie 14×14  ← on s'arrête ici
        #   [28]     → MaxPool du bloc 4 → 7×7 (on l'exclut pour garder 14×14)
        self.features = nn.Sequential(*list(vgg19.features.children())[:29])

        # Par défaut les poids sont gelés (pas de gradient CNN pendant l'entraînement).
        # Le fine-tuning peut être activé via set_fine_tune(True) après N epochs.
        self.set_fine_tune(fine_tune)

        self.encoded_dim = encoded_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Passe avant : extrait les feature maps et les reformate en séquence.

        Args:
            images: (B, 3, 224, 224)

        Returns:
            features: (B, 196, 512)
        """
        out = self.features(images)    # (B, 512, 14, 14) — feature maps spatiales

        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1)  # (B, 14, 14, 512) — canaux en dernier
        out = out.reshape(B, H * W, C) # (B, 196, 512)    — aplatit la grille en séquence

        # Chaque vecteur out[b, i, :] est un "annotation vector" aᵢ du papier.
        return out

    def set_fine_tune(self, fine_tune: bool) -> None:
        """Active ou gèle les gradients des poids CNN."""
        for param in self.features.parameters():
            param.requires_grad = fine_tune
