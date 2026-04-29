import torch
import torch.nn as nn
from src.models.attention import Attention
from src.dataset.vocabulary import Vocabulary


class Decoder(nn.Module):
    def __init__(
        self,
        device,
        encoder_dim=512,
        vocab_size=100,
        max_seq_len=124,
        dropout_prob=0.3,
        use_tf=False,
        
    ):
        super().__init__()
        self.vocab = Vocabulary.load("data/flicker8k/vocab.json")
        self.use_tf = use_tf
        self.device = device
        
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim

        self.embedding = nn.Embedding(vocab_size, 512)
        self.h_mlp = nn.Linear(encoder_dim, 512)
        self.c_mlp = nn.Linear(encoder_dim, 512)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.attention = Attention(encoder_dim)

        self.dropout = nn.Dropout(p=dropout_prob)
        self.lstm_cell = nn.LSTMCell(512 + encoder_dim, 512)

        self.L_o = nn.Linear(512, vocab_size)
        self.L_h = nn.Linear(512, 512)
        self.L_z = nn.Linear(512, 512)

    def forward(self, annot_vecs, captions):
        self.h_0, self.c_0 = self._init_h_c(annot_vecs)
        
        batch_size = annot_vecs.shape[0]
        
        prev_words = torch.ones(batch_size, 1).long().to(self.device)
        
        if self.use_tf:
            embeddings = self.embedding(captions)
        else:
            embeddings = self.embedding(prev_words)

        # for _ in range(self.max_seq_len):
        #     output = None

    def _init_h_c(self, annot_vecs):
        annot_mean = torch.mean(annot_vecs, dim=1)
        h_0 = self.tanh(self.h_mlp(annot_mean))
        c_0 = self.tanh(self.c_mlp(annot_mean))
        return h_0, c_0
