"""Normalization functionality."""
import torch
from torch import nn, Tensor

from small_transformer.base import ModelBase


class LayerNorm(ModelBase):
    """Z-normalization."""
    def __init__(self, d_model, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias

        return out


class RMSNorm(ModelBase):
    """RMSNorm.

    Re-scaling the activations with only the root mean square (RMS)
    of the summed activations.

    Reference:
    Root mean square layer normalization by B. Zhang and R. Sennrich (2019).
    """
    def __init__(self, in_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(in_features))

    def get_norm(self, x: Tensor):
        """Calculate the norm."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor):
        normed = self.get_norm(x.float())
        return normed.type_as(x)
