
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU Feed Forward Network

    Input:
        (batch, seq_len, d_model)

    Output:
        (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()

        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):

        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)

        x = gate * value

        x = self.down_proj(x)

        return x
