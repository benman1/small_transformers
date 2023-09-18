"""Feedforward network."""
from typing import Callable

from torch import nn as nn, Tensor
from torch.nn import functional as F

from small_transformer.base import ModelBase
from small_transformer.model import log


class FeedForwardNetwork(ModelBase):
    def __init__(self, in_features: int, out_features: int, act_fun: Callable = F.gelu):
        """Init.

        Use GLU gating by default.
        Reference: GLU Variants Improve Transformer, Noam Shazeer (2020).
        """
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.act_fun = act_fun
        log.info(f'<init>: \n{self}')

    def forward(self, x: Tensor) -> Tensor:
        """
        Use GLU gating by default.
        """
        x = self.linear1(x)
        x = self.act_fun(x)
        return x
