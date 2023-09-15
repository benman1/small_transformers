"""Normalization functionality."""
import torch
from torch import nn

from small_transformer.base import ModelBase


class LayerNorm(ModelBase):
    """Z-normalization.

    Gamma and beta are learnable parameters that scale and shift.
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta

        return out


class RMSNorm(ModelBase):
    """RMSNorm.

    Re-scaling the activations with only the root mean square (RMS)
    of the summed activations.

    Reference:
    Root mean square layer normalization by B. Zhang and R. Sennrich (2019).
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).sum(-1, keepdim=True).sqrt()
        x = x / (rms + self.eps)
        return x
