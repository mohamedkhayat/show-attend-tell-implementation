import logging

import torch
import torch.nn as nn
from src.models.attention import Attention
from src.dataset.vocabulary import Vocabulary
from src.utils.logging import configure_logging

logger = logging.getLogger(__name__)


class Decoder(nn.Module):
    def __init__(
        self,
        device,
        encoder_dim=512,
        max_seq_len=124,
        dropout_prob=0.3,
        use_tf=False,
    ):
        super().__init__()
        self.vocab = Vocabulary.load("data/flicker8k/vocab.json")

        self.use_tf = use_tf
        self.device = device

        self.max_seq_len = max_seq_len
        self.vocab_size = len(self.vocab.word2idx)
        self.encoder_dim = encoder_dim
        
        self.b_mlp = nn.Linear(encoder_dim, 1)

        self.embedding = nn.Embedding(self.vocab_size, 512)
        self.h_mlp = nn.Linear(encoder_dim, 512)
        self.c_mlp = nn.Linear(encoder_dim, 512)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.attention = Attention(encoder_dim)

        self.dropout = nn.Dropout(p=dropout_prob)
        self.lstm_cell = nn.LSTMCell(512 + encoder_dim, 512)

        self.L_o = nn.Linear(512, self.vocab_size)
        self.L_h = nn.Linear(512, 512)
        self.L_z = nn.Linear(512, 512)

    def forward(self, annot_vecs, captions):
        # DONT FORGET DEEP OUTPUT !
        # MOMKEN ADD STOCHASTIC GATED DRA CHNIA !
        h, c = self._init_h_c(annot_vecs)

        batch_size = annot_vecs.shape[0]

        prev_words = torch.ones(batch_size, 1).long().to(self.device)

        if self.use_tf:
            embeddings = (
                self.embedding(captions)
                if self.training
                else self.embedding(prev_words)
            )
        else:
            embedding = self.embedding(prev_words)

        logger.debug("word count shape : %s", self.vocab_size)
        preds = torch.zeros(batch_size, self.max_seq_len, self.vocab_size)
        alpha_scores = torch.zeros(batch_size, self.max_seq_len, annot_vecs.size(1))

        for t in range(self.max_seq_len):
            logger.debug(f" time step : {t}")
            context, alpha = self.attention(annot_vecs, h)
            if self.use_tf and self.training:
                embedding = embeddings[:, t]
            else:
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding

            logger.debug(
                "use_tf ? : %s, embedding shape : %s", self.use_tf, embedding.shape
            )
            logger.debug(f"context shape : {context.shape}")
            lstm_input = torch.cat([embedding, context], dim=1)

            logger.debug("lstm_input shape : %s", lstm_input.shape)

            h, c = self.lstm_cell(lstm_input, (h, c))

            context_proj = self.L_z(context)

            hidden_proj = self.L_h(h)

            deep_output = self.L_o(embedding + context_proj + hidden_proj)
            output = self.dropout(deep_output)

            logger.info(f"Decoder output shape : {output.shape}")

            preds[:, t] = output
            alpha_scores[:, t] = alpha

            if not self.training or not self.use_tf:
                prev_word = torch.argmax(output, dim=1).reshape((batch_size, 1))
                logger.info("prev word shape : %s", prev_word.shape)
                embedding = self.embedding(prev_word)
                logger.debug(f"embedding shape : {embedding.shape}")

        return preds, alpha_scores

    def _init_h_c(self, annot_vecs):
        annot_mean = torch.mean(annot_vecs, dim=1)
        h_0 = self.tanh(self.h_mlp(annot_mean))
        c_0 = self.tanh(self.c_mlp(annot_mean))
        return h_0, c_0


if __name__ == "__main__":
    configure_logging("INFO")
    vocab = Vocabulary.load("data/flicker8k/vocab.json")
    vocab_size = len(vocab.word2idx)
    device = torch.device("cuda")
    use_tf = False
    model = Decoder(device, use_tf=use_tf).to(device)
    captions = torch.zeros(32, 124).long().to(device)
    inp = torch.zeros((32, 196, 512)).to(device)
    model.train()
    out = model(inp, captions)
