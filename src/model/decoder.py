import torch
import torch.nn as nn

from src.model.attention import SoftAttention


class Decoder(nn.Module):
    """LSTM decoder with soft attention (Sec. 3.1.2 & 4.2 of the paper).

    At each time step t:
      1. Attention computes context vector z_hat from encoder features and h_{t-1}
      2. A gating scalar beta modulates the visual context
      3. LSTMCell updates hidden state using (word_embed, h_{t-1}, beta * z_hat)
      4. Deep output layer predicts next word distribution

    Forward returns logits for all time steps and all attention weights,
    so the training loop can compute cross-entropy loss and doubly stochastic
    regularization in one pass.
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

        self.encoder_dim  = encoder_dim
        self.decoder_dim  = decoder_dim
        self.embed_dim    = embed_dim
        self.vocab_size   = vocab_size

        # Word embedding  E ∈ ℝ^{m × K}
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout   = nn.Dropout(dropout)

        # Attention module
        self.attention = SoftAttention(encoder_dim, decoder_dim, attention_dim)

        # Gating scalar β_t = σ(f_β(h_{t-1}))
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        # LSTM cell — input = embed + context
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        # Initialise h0 and c0 from mean annotation vector (Sec. 3.1.2)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Deep output layer: Lo · (E·y_{t-1} + Lh·h_t + Lz·z_hat)
        self.L_o = nn.Linear(embed_dim, vocab_size)
        self.L_h = nn.Linear(decoder_dim, embed_dim)
        self.L_z = nn.Linear(encoder_dim, embed_dim)

    # ------------------------------------------------------------------
    def _init_hidden(self, features: torch.Tensor):
        """h0, c0 from mean of annotation vectors."""
        mean_feat = features.mean(dim=1)              # (B, encoder_dim)
        h = torch.tanh(self.init_h(mean_feat))        # (B, decoder_dim)
        c = torch.tanh(self.init_c(mean_feat))        # (B, decoder_dim)
        return h, c

    # ------------------------------------------------------------------
    def forward(
        self,
        features: torch.Tensor,
        captions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forcing forward pass (used during training).

        Args:
            features : (B, L, encoder_dim)  — encoder annotation vectors
            captions : (B, T)               — ground-truth token indices
                       first token is <start>, last is excluded from input
                       (we predict tokens 1..T from inputs 0..T-1)

        Returns:
            logits  : (B, T-1, vocab_size)  — unnormalised word scores
            alphas  : (B, T-1, L)           — attention weights per step
        """
        B = features.size(0)
        T = captions.size(1)

        embeds = self.dropout(self.embedding(captions))   # (B, T, embed_dim)

        h, c = self._init_hidden(features)

        logits_list = []
        alphas_list = []

        # Feed tokens 0..T-2 as input, predict tokens 1..T-1
        for t in range(T - 1):
            z_hat, alpha = self.attention(features, h)    # (B, enc_dim), (B, L)

            # Gating scalar
            beta  = torch.sigmoid(self.f_beta(h))         # (B, encoder_dim)
            z_hat = beta * z_hat                          # (B, encoder_dim)

            lstm_input = torch.cat([embeds[:, t, :], z_hat], dim=1)  # (B, embed+enc)
            h, c = self.lstm_cell(lstm_input, (h, c))

            # Deep output: combine word embed, hidden state, context
            out = self.L_o(
                self.dropout(
                    embeds[:, t, :] + self.L_h(h) + self.L_z(z_hat)
                )
            )                                             # (B, vocab_size)

            logits_list.append(out)
            alphas_list.append(alpha)

        logits = torch.stack(logits_list, dim=1)          # (B, T-1, vocab_size)
        alphas = torch.stack(alphas_list, dim=1)          # (B, T-1, L)

        return logits, alphas
