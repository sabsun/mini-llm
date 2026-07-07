
import torch
import torch.nn as nn

from rmsnorm import RMSNorm
from attention import MultiHeadAttention
from swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """
    Mini-LLaMA Transformer Block (Pre-Norm)

    Input:
        (batch, seq_len, d_model)

    Output:
        (batch, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.attention_norm = RMSNorm(d_model)

        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
        )

        self.ffn_norm = RMSNorm(d_model)

        self.ffn = SwiGLU(
            d_model=d_model,
            hidden_dim=hidden_dim,
        )

    def forward(self, x):

        # Attention block
        x = x + self.attention(
            self.attention_norm(x)
        )

        # Feed Forward block
        x = x + self.ffn(
            self.ffn_norm(x)
        )

        return x
