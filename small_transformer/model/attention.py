"""Attention mechanisms.

Attention mechanisms are essential components of Transformers,
and full attention, as well as sparse attention, can be employed.
"""
from typing import Literal

import torch
from torch import Tensor, softmax
from torch import nn as nn

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
            attention_type: str = Literal["Multihead", "Softmax", "Flash"]
    ):
        """Init.
        Args:
            in_features (int): Embedding dimensionality.
            n_heads (int): Number of attention heads (can be used for normalization).
            max_seq_len (int): Maximum relative position.
            attention_type (str): Which attention to use

        TODO: implement positional encoding?
        """
        super().__init__()

        self.in_features = in_features
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.d_k = in_features // self.n_heads
        self.attention_type = attention_type
        self.qkv_proj = None

        # weight matrixes of the same size as the transformer model's
        # token embedding dimensionality
        self.qkv_proj = nn.Linear(
            in_features=self.in_features,
            out_features=3 * self.in_features
        )

        match self.attention_type:
            case "Multihead":
                self.out_proj = nn.Linear(self.in_features, self.in_features)
            case _:
                if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    raise ValueError("Torch implementation too old!")

                mask = torch.full((1, 1, self.max_seq_len, self.max_seq_len), float("-inf"))
                mask = torch.triu(mask, diagonal=1)
                self.register_buffer("mask", mask)

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
        # flash implementation
        if self.qkv_proj is None:
            return torch.nn.functional.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )

        scores = Q @ K.transpose(-2, -1)
        # softmax to keep gradients stable:
        attention = softmax(scores / K.shape[1] ** 0.5, dim=-1)
        # weighted sum of all four value vectors:
        context = (attention @ V)
        context = context.transpose(1, 2).reshape(
            context.shape[0], context.shape[1], self.in_features
        )
        if self.attention_type != "Multihead":
            return context

        return self.out_proj(context)
