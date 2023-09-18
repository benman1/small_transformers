"""Attention mechanisms.

Attention mechanisms are essential components of Transformers,
and full attention, as well as sparse attention, can be employed.
"""

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
    ):
        """Init.
        Args:
            in_features (int): Embedding dimensionality.
            n_heads (int): Number of attention heads (can be used for normalization).
            max_seq_len (int): Maximum relative position.

        TODO: implement positional encoding?
        """
        super().__init__()

        self.in_features = in_features
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.d_k = in_features // self.n_heads
        self.qkv_proj = None

        # weight matrixes of the same size as the transformer model's
        # token embedding dimensionality
        self.qkv_proj = nn.Linear(
            in_features=self.in_features,
            out_features=3 * self.in_features
        )
        # mask = torch.full((1, 1, self.max_seq_len, self.max_seq_len), float("-inf"))
        # mask = torch.triu(mask, diagonal=1)
        # self.register_buffer("mask", mask)

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
        scores = Q @ K.transpose(-2, -1)
        # assert hasattr(self, 'mask')
        # scores = scores + self.mask[:, :, :self.max_seq_len, :self.max_seq_len]
        # softmax to keep gradients stable:
        attention = softmax(scores / K.shape[1] ** 0.5, dim=-1)
        # weighted sum of all four value vectors:
        context = (attention @ V)
        context = context.transpose(1, 2).reshape(
            context.shape[0], context.shape[1], self.in_features
        )
        return context


class MultiHeadAttention(Attention):
    def __init__(
            self,
            in_features: int,
            n_heads: int,
            max_seq_len: int = 100,
    ):
        """Init.
        Args:
            in_features (int): Embedding dimensionality.
            n_heads (int): Number of attention heads (can be used for normalization).
            max_seq_len (int): Maximum relative position.
        """
        super().__init__(in_features, n_heads, max_seq_len)
        self.out_proj = nn.Linear(self.in_features, self.in_features)

    def forward(self, x: Tensor) -> Tensor:
        """Apply multihead attention."""
        context = super().forward(x)
        return self.out_proj(context)


class FlashAttention(Attention):
    """Implementation of Flash Attention."""

    def __init__(
            self,
            in_features: int,
            n_heads: int,
            max_seq_len: int = 100,
            dropout: float = 0.1
    ):
        """Init.
        Args:
            in_features (int): Embedding dimensionality.
            n_heads (int): Number of attention heads (can be used for normalization).
            max_seq_len (int): Maximum relative position.
            dropout: dropout ratio for flash attention.
        """
        super().__init__(in_features, n_heads, max_seq_len)
        if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            raise RuntimeError("Torch implementation too old!")
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        """Run flash attention."""
        batch_size, seq_length, _ = x.shape
        Q, K, V = self.project_qkv(x)
        output = torch.nn.functional.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        return output
