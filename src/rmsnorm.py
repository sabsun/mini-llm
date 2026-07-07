import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Used in LLaMA models.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):

        rms = torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )

        return x * rms * self.weight