import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAttention(nn.Module):
    """Deterministic soft attention (Sec. 4.2 of the paper).

    For each decoder step t, computes a context vector z_hat as a weighted
    sum of the encoder annotation vectors:

        e_ti   = W2 · tanh(W1_a · a_i + W1_h · h_{t-1})
        alpha  = softmax(e_t)                               # (B, L)
        z_hat  = sum_i alpha_i * a_i                        # (B, D)

    Also returns alpha so the training loop can apply doubly stochastic
    regularization: λ · Σ_i (1 - Σ_t α_ti)²
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        """
        Args:
            encoder_dim:   D — dimension of annotation vectors (512 for VGG19)
            decoder_dim:   n — dimension of LSTM hidden state
            attention_dim: intermediate projection dimension
        """
        super().__init__()

        self.W1_a = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W1_h = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.W2   = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features : (B, L, encoder_dim)  — annotation vectors aᵢ
            hidden   : (B, decoder_dim)     — LSTM hidden state h_{t-1}

        Returns:
            z_hat : (B, encoder_dim)  — context vector
            alpha : (B, L)            — attention weights (sum to 1)
        """
        # Project features and hidden state into attention space
        att_features = self.W1_a(features)               # (B, L, attention_dim)
        att_hidden   = self.W1_h(hidden).unsqueeze(1)    # (B, 1, attention_dim)

        # Score each location
        e = self.W2(torch.tanh(att_features + att_hidden)).squeeze(2)  # (B, L)

        alpha = F.softmax(e, dim=1)                      # (B, L)

        # Weighted sum of annotation vectors
        z_hat = (alpha.unsqueeze(2) * features).sum(dim=1)  # (B, encoder_dim)

        return z_hat, alpha
