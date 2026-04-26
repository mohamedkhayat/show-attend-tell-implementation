# =============================================================================
# attention.py — Mécanisme d'attention douce (Soft Attention)
#
# Rôle : à chaque pas de temps t du décodeur, décider QUELLES régions de
# l'image regarder pour générer le prochain mot.
#
# Principe du papier (Sec. 4.2 — Deterministic "Soft" Attention) :
#   Le décodeur LSTM a un état caché h_{t-1}. On compare cet état avec chacun
#   des 196 annotation vectors (régions de l'image) via un petit MLP de scoring.
#   Le résultat est un vecteur de poids αₜ (somme = 1) qui indique l'importance
#   de chaque région. Le contexte visuel ẑₜ est la somme pondérée des régions.
#
# Formules (Eq. 4 & 5 du papier) :
#   eₜᵢ = W2 · tanh(W1_a · aᵢ + W1_h · hₜ₋₁)   ← score de chaque région i
#   αₜ  = softmax(eₜ)                              ← poids normalisés (somme = 1)
#   ẑₜ  = Σᵢ αₜᵢ · aᵢ                             ← contexte visuel pour le LSTM
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAttention(nn.Module):
    """Attention douce déterministe (Sec. 4.2 du papier).

    Entrées : annotation vectors (B, L, encoder_dim) + état caché LSTM (B, decoder_dim)
    Sorties : contexte visuel ẑₜ (B, encoder_dim) + poids αₜ (B, L)
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        """
        Args:
            encoder_dim   : dimension des annotation vectors aᵢ (512 pour VGG19)
            decoder_dim   : dimension de l'état caché LSTM hₜ₋₁
            attention_dim : dimension intermédiaire du MLP de scoring
        """
        super().__init__()

        # Projette chaque annotation vector dans l'espace d'attention
        self.W1_a = nn.Linear(encoder_dim, attention_dim, bias=False)

        # Projette l'état caché du LSTM dans le même espace d'attention
        self.W1_h = nn.Linear(decoder_dim, attention_dim, bias=False)

        # Produit un score scalaire par région (B, L, attention_dim) → (B, L, 1)
        self.W2 = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features : (B, L, encoder_dim)  — les 196 annotation vectors aᵢ
            hidden   : (B, decoder_dim)     — état caché hₜ₋₁ du LSTM

        Returns:
            z_hat : (B, encoder_dim)  — contexte visuel ẑₜ (somme pondérée)
            alpha : (B, L)            — poids d'attention (somme = 1 sur L)
        """
        att_features = self.W1_a(features)            # (B, L, attention_dim) — contribution de chaque région
        att_hidden   = self.W1_h(hidden).unsqueeze(1) # (B, 1, attention_dim) — broadcast sur les L régions

        # Addition + tanh + projection → score scalaire par région
        e = self.W2(torch.tanh(att_features + att_hidden)).squeeze(2)  # (B, L)

        # Normalisation : αₜ somme à 1 → interprétable comme distribution de probabilité
        alpha = F.softmax(e, dim=1)  # (B, L)

        # Contexte visuel : somme pondérée des annotation vectors
        # alpha.unsqueeze(2) : (B, L, 1) × features : (B, L, enc_dim) → (B, L, enc_dim)
        z_hat = (alpha.unsqueeze(2) * features).sum(dim=1)  # (B, encoder_dim)

        # alpha est retourné pour la doubly stochastic regularization dans train.py :
        # λ · Σᵢ(1 - Σₜ αₜᵢ)²  → force le modèle à regarder toutes les régions au total
        return z_hat, alpha
