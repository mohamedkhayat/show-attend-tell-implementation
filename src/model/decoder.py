# =============================================================================
# decoder.py — Décodeur LSTM avec attention douce
#
# Rôle : générer une légende mot par mot en s'appuyant sur les annotation
# vectors de l'encodeur et en "regardant" les bonnes régions à chaque étape.
#
# Principe du papier (Sec. 3.1.2 & 4.2) :
#   À chaque pas t, le LSTM reçoit :
#     - l'embedding du mot précédent  E·yₜ₋₁
#     - le contexte visuel ẑₜ (fourni par l'attention)
#   et produit un état caché hₜ qui sert à prédire le prochain mot via une
#   couche de sortie "deep output" combinant mot précédent, hₜ et ẑₜ.
#
#   Initialisation (Sec. 3.1.2) :
#     h₀ = tanh(MLP(mean(aᵢ)))   ← moyenne des annotation vectors
#     c₀ = tanh(MLP(mean(aᵢ)))
#
#   Gating scalar βₜ (Sec. 4.2.1) :
#     βₜ = σ(f(hₜ₋₁))  — module l'importance du contexte visuel à chaque étape
#
#   Deep output (Eq. 7 du papier) :
#     p(yₜ) ∝ exp(Lo · (E·yₜ₋₁ + Lh·hₜ + Lz·ẑₜ))
# =============================================================================

import torch
import torch.nn as nn

from src.model.attention import SoftAttention


class Decoder(nn.Module):
    """Décodeur LSTM avec soft attention (Sec. 3.1.2 & 4.2 du papier).

    Mode entraînement : teacher forcing — on fournit le vrai token à chaque étape.
    Mode inférence    : voir inference.py — décodage greedy token par token.

    Forward retourne logits (B, T-1, vocab_size) et alphas (B, T-1, L)
    pour que la boucle d'entraînement calcule la loss en un seul passage.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int = 512,
        embed_dim: int = 512,
        decoder_dim: int = 512,
        attention_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.embed_dim   = embed_dim
        self.vocab_size  = vocab_size

        # Embedding des mots : convertit un index en vecteur dense de dim embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout   = nn.Dropout(dropout)

        # Module d'attention : calcule ẑₜ et αₜ à partir des features et de hₜ₋₁
        self.attention = SoftAttention(encoder_dim, decoder_dim, attention_dim)

        # Gating scalar : βₜ = σ(f(hₜ₋₁)) — module l'importance du contexte visuel
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        # LSTM cell : reçoit (embed + contexte visuel) et produit (hₜ, cₜ)
        # Taille d'entrée = embed_dim + encoder_dim (concaténation)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        # Initialisation de h₀ et c₀ depuis la moyenne des annotation vectors
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Deep output layer (Eq. 7) : combine embedding, hₜ et ẑₜ pour prédire le mot
        self.L_o = nn.Linear(embed_dim, vocab_size)   # projection finale vers vocab
        self.L_h = nn.Linear(decoder_dim, embed_dim)  # contribution de hₜ
        self.L_z = nn.Linear(encoder_dim, embed_dim)  # contribution de ẑₜ

    def _init_hidden(self, features: torch.Tensor):
        """Initialise h₀ et c₀ à partir de la moyenne des annotation vectors.

        Fidèle à la Sec. 3.1.2 du papier : h₀ = tanh(W_h · mean(aᵢ))
        """
        mean_feat = features.mean(dim=1)         # (B, encoder_dim) — moyenne sur les 196 régions
        h = torch.tanh(self.init_h(mean_feat))   # (B, decoder_dim)
        c = torch.tanh(self.init_c(mean_feat))   # (B, decoder_dim)
        return h, c

    def forward(
        self,
        features: torch.Tensor,
        captions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Passe avant en teacher forcing (utilisée pendant l'entraînement).

        On fournit le vrai token yₜ₋₁ à chaque étape (pas la prédiction),
        ce qui accélère l'apprentissage et stabilise l'entraînement.

        Args:
            features : (B, L, encoder_dim)  — annotation vectors de l'encodeur
            captions : (B, T)               — séquences de tokens ground-truth
                       token[0] = <start>, token[T-1] = <end> ou <pad>

        Returns:
            logits  : (B, T-1, vocab_size)  — scores non-normalisés pour chaque position
            alphas  : (B, T-1, L)           — poids d'attention pour la régularisation
        """
        B = features.size(0)
        T = captions.size(1)

        # Embedding de toute la séquence d'entrée en une fois (plus efficace)
        embeds = self.dropout(self.embedding(captions))  # (B, T, embed_dim)

        h, c = self._init_hidden(features)

        logits_list = []
        alphas_list = []

        # On prédit les tokens 1..T-1 à partir des entrées 0..T-2
        # (on ne prédit pas <start>, on ne consomme pas <end> comme entrée)
        for t in range(T - 1):
            z_hat, alpha = self.attention(features, h)   # contexte visuel du pas t

            # Gating : βₜ module l'importance du contexte visuel (Sec. 4.2.1)
            beta  = torch.sigmoid(self.f_beta(h))        # (B, encoder_dim) — valeurs entre 0 et 1
            z_hat = beta * z_hat                         # (B, encoder_dim)

            # Entrée LSTM : concaténation du mot courant et du contexte visuel
            lstm_input = torch.cat([embeds[:, t, :], z_hat], dim=1)  # (B, embed+enc)
            h, c = self.lstm_cell(lstm_input, (h, c))

            # Deep output (Eq. 7) : somme des trois contributions avant projection
            out = self.L_o(
                self.dropout(
                    embeds[:, t, :] + self.L_h(h) + self.L_z(z_hat)
                )
            )  # (B, vocab_size) — logits du pas t

            logits_list.append(out)
            alphas_list.append(alpha)

        logits = torch.stack(logits_list, dim=1)  # (B, T-1, vocab_size)
        alphas = torch.stack(alphas_list, dim=1)  # (B, T-1, L) — pour la régularisation

        return logits, alphas
