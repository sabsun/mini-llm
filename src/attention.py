import math
import torch
import torch.nn as nn

from rope import RotaryEmbedding


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):

        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim)

    def _split_heads(self, x):

        batch, seq, _ = x.shape

        x = x.view(
            batch,
            seq,
            self.n_heads,
            self.head_dim
        )

        return x.transpose(1,2)

    def _merge_heads(self, x):

        batch, heads, seq, dim = x.shape

        x = x.transpose(1,2)

        return x.reshape(
            batch,
            seq,
            self.d_model
        )

    def forward(self, x):

        q = self._split_heads(
            self.q_proj(x)
        )

        k = self._split_heads(
            self.k_proj(x)
        )

        v = self._split_heads(
            self.v_proj(x)
        )

        q, k = self.rope(q, k)

        scores = torch.matmul(
            q,
            k.transpose(-2,-1)
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        seq_len = scores.size(-1)

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            ),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        weights = torch.softmax(
            scores,
            dim=-1
        )

        output = torch.matmul(
            weights,
            v
        )

        output = self._merge_heads(
            output
        )

        output = self.out_proj(
            output
        )

        return output