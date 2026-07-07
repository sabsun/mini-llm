
import torch
import torch.nn as nn

from embeddings import TokenEmbedding
from rmsnorm import RMSNorm
from block import TransformerBlock


class MiniLlama(nn.Module):
    """
    Mini-LLaMA v1.0

    Input:
        token_ids
            (batch, seq_len)

    Output:
        logits
            (batch, seq_len, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.embedding = TokenEmbedding(
            vocab_size,
            d_model,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    hidden_dim=hidden_dim,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = RMSNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

    def forward(self, token_ids):

        x = self.embedding(token_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits
