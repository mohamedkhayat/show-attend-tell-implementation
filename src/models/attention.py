import logging

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class Attention(nn.Module):
    def __init__(self, encoder_dim=512):
        super().__init__()
        self.W_h = nn.Linear(512, 512)
        self.W_a = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)

    def forward(self, an_vec, prev_hidden):
        logger.debug(
            "Attention input shapes | encoder=%s hidden=%s",
            tuple(an_vec.shape),
            tuple(prev_hidden.shape),
        )
        hidden_proj = self.W_h(prev_hidden).unsqueeze(1)
        ann_proj = self.W_a(an_vec)
        logger.debug(
            "Attention projections | encoder=%s hidden=%s",
            tuple(ann_proj.shape),
            tuple(hidden_proj.shape),
        )

        energy = self.v(torch.tanh(hidden_proj + ann_proj)).squeeze(2)
        logger.debug("Attention energy shape | %s", tuple(energy.shape))
        attention_scores = torch.softmax(energy, dim=1)  # shape : (B, L)
        logger.debug("Attention score shape | %s", tuple(attention_scores.shape))
        # an_vec shape : (B, L, D)
        context = (attention_scores.unsqueeze(2) * an_vec).sum(dim=1)
        logger.debug("Context shape | %s", tuple(context.shape))
        return context, attention_scores
