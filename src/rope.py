
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE)

    Input:
        q : (batch, heads, seq_len, head_dim)
        k : (batch, heads, seq_len, head_dim)

    Output:
        q_rot, k_rot : same shape
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even.")

        self.head_dim = head_dim
        self.base = base

    def _rotate_half(self, x):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        return torch.stack(
            (-x_odd, x_even),
            dim=-1
        ).flatten(-2)

    def forward(self, q, k):

        device = q.device

        seq_len = q.shape[-2]

        half_dim = self.head_dim // 2

        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(
                    half_dim,
                    device=device,
                    dtype=torch.float32
                )
                / half_dim
            )
        )

        positions = torch.arange(
            seq_len,
            device=device,
            dtype=torch.float32
        )

        angles = torch.outer(
            positions,
            inv_freq
        )

        cos = torch.repeat_interleave(
            angles.cos(),
            2,
            dim=-1
        )

        sin = torch.repeat_interleave(
            angles.sin(),
            2,
            dim=-1
        )

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin

        return q, k
