import logging

import torch
import torch.nn as nn
from src.models.attention import Attention
from src.dataset.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class Decoder(nn.Module):
    def __init__(
        self,
        device,
        encoder_dim=512,
        max_seq_len=40,
        dropout_prob=0.5,
        use_tf=True,
    ):
        super().__init__()
        self.vocab = Vocabulary.load("data/flicker8k/vocab.json")

        self.use_tf = use_tf
        self.device = device
        self.max_seq_len = max_seq_len
        self.vocab_size = len(self.vocab.word2idx)
        self.encoder_dim = encoder_dim

        self.embedding = nn.Embedding(self.vocab_size, 512)
        self.h_mlp = nn.Linear(encoder_dim, 512)
        self.c_mlp = nn.Linear(encoder_dim, 512)

        self.attention = Attention(encoder_dim)

        # Gating scalar β_t = σ(f_β(h_{t-1})) — section 4.2 of the paper
        self.beta_gate = nn.Linear(512, 1)

        self.dropout = nn.Dropout(p=dropout_prob)
        self.lstm_cell = nn.LSTMCell(512 + encoder_dim, 512)

        # Deep output: L_o(E*y_{t-1} + L_h*h_t + L_z*z_t)
        self.L_o = nn.Linear(512, self.vocab_size)
        self.L_h = nn.Linear(512, 512)
        self.L_z = nn.Linear(encoder_dim, 512)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, annot_vecs, captions=None):
        """
        Training (use_tf=True): captions (B, T) drives teacher-forced unrolling for T-1 steps.
        Inference (captions=None): greedy decode for max_seq_len steps.
        Returns preds (B, n_steps, vocab_size) and alphas (B, n_steps, L).
        """
        h, c = self._init_h_c(annot_vecs)
        B = annot_vecs.size(0)
        L = annot_vecs.size(1)

        use_tf = self.use_tf and self.training and captions is not None

        if use_tf:
            T = captions.size(1)
            n_steps = T - 1  # predict T-1 words (skip <start> in targets)
            all_embs = self.embedding(captions)  # (B, T, emb_dim)
            embedding = all_embs[:, 0]           # embed(<start>)
        else:
            n_steps = self.max_seq_len
            embedding = self.embedding(
                torch.ones(B, dtype=torch.long, device=self.device)  # <start>=1
            )

        preds = torch.zeros(B, n_steps, self.vocab_size, device=self.device)
        alphas = torch.zeros(B, n_steps, L, device=self.device)

        for t in range(n_steps):
            # h here is h_{t-1}; attention and gating use it before LSTM update
            context, alpha = self.attention(annot_vecs, h)
            beta = self.sigmoid(self.beta_gate(h))  # (B, 1)
            context = beta * context                 # gated context z_t

            h, c = self.lstm_cell(
                torch.cat([embedding, context], dim=1), (h, c)
            )

            # Deep output layer
            out = self.L_o(
                self.dropout(embedding + self.L_h(h) + self.L_z(context))
            )

            preds[:, t] = out
            alphas[:, t] = alpha

            # Prepare next embedding
            if use_tf and t + 1 < n_steps:
                embedding = all_embs[:, t + 1]
            elif not use_tf:
                embedding = self.embedding(out.argmax(dim=1))

        return preds, alphas

    def _init_h_c(self, annot_vecs):
        mean = annot_vecs.mean(dim=1)
        h0 = self.tanh(self.h_mlp(mean))
        c0 = self.tanh(self.c_mlp(mean))
        return h0, c0
