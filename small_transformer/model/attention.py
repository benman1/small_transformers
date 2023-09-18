"""Attention mechanisms.

Attention mechanisms are essential components of Transformers,
and full attention, as well as sparse attention, can be employed.
"""
import math

import torch
from torch import nn as nn, Tensor, softmax

from small_transformer.base import ModelBase
from small_transformer.utils import setup_logger

log = setup_logger(__name__)


class Attention(ModelBase):
    """Relative positional encoding attention.

    Adds relative positional encodings to the queries before applying attention.
    """

    def __init__(
            self,
            in_features: int,
            n_heads: int,
            max_seq_len: int = 100,
            multi_head: bool = False
    ):
        """Init.
        Args:
            in_features (int): Embedding dimensionality.
            n_heads (int): Number of attention heads (can be used for normalization).
            max_seq_len (int): Maximum relative position.
            multi_head (bool): Whether to apply output projection (MHA).

        TODO: implement positional encoding?
        """
        super().__init__()

        self.in_features = in_features
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.d_k = in_features // self.n_heads

        # weight matrixes of the same size as the transformer model's
        # token embedding dimensionality
        self.qkv_proj = nn.Linear(
            in_features=self.in_features,
            out_features=3 * self.in_features
        )
        self.multi_head = multi_head
        if self.multi_head:
            self.out_proj = nn.Linear(self.in_features, self.in_features)
        else:
            self.out_proj = None

        log.info(f'<init>: \n{self}')

    def project_qkv(self, x: Tensor) -> (Tensor, Tensor, Tensor):
        """Queries, keys, and value projections."""

        def _reshape(x: Tensor) -> Tensor:
            """Reshape as needed."""
            return x.reshape(x.shape[0], x.shape[1], self.n_heads, self.d_k)

        qkv = self.qkv_proj(x)
        Q, K, V = torch.chunk(qkv, 3, dim=-1)
        return _reshape(Q), _reshape(K), _reshape(V)

    def forward(self, x: Tensor):
        """Add relative attention.

        Add relative positional embeddings to the queries.
        """
        Q, K, V = self.project_qkv(x)
        scores = Q @ K.transpose(-2, -1)
        # softmax to keep gradients stable:
        attention = softmax(scores / K.shape[1] ** 0.5, dim=-1)
        # weighted sum of all four value vectors:
        context = (attention @ V)
        context = context.transpose(1, 2).reshape(
            context.shape[0], context.shape[1], self.in_features
        )
        if not self.multi_head:
            return context

        return self.out_proj(context)
