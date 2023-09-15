"""Attention mechanisms.

Attention mechanisms are essential components of Transformers,
and full attention, as well as sparse attention, can be employed.
"""
from abc import abstractmethod

import torch
from torch import nn as nn
from torch.nn import functional as F

from small_transformer.base import ModelBase
from small_transformer.utils import setup_logger


log = setup_logger(__name__)


class Attention(ModelBase):
    """Abstract Attention class handles projections.

    _calculate_attention() defined abstractly.
    Attention mechanisms override this with specific logic.
    """
    def __init__(self, d_model, n_heads: int):
        super().__init__()

        self.n_heads = n_heads
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        log.info(f'<init>: \n{self}')

    def forward(self, q, k, v, mask=None):
        # Apply linear projections
        q = self.q_lin(q)
        k = self.k_lin(k)
        v = self.v_lin(v)

        # Calculate attention
        attn_out = self._calculate_attention(q, k, v, mask)

        return attn_out

    @abstractmethod
    def _calculate_attention(self, q, k, v, mask):
        pass


class FlashAttention(Attention):
    """FlashAttention mechanism.

    This aims to optimize the speed and memory consumption of
    attention modules on GPUs. FlashAttention organizes the
    input into blocks and introduces recomputation to make better
    use of the fast memory on GPUs. It has been integrated into
    frameworks like PyTorch, DeepSpeed, and Megatron-LM.
    """
    def __init__(self, d_model, n_heads: int, n_chunks: int):
        super().__init__(d_model, n_heads)
        self.n_chunks = n_chunks

    def _calculate_attention(self, q, k, v, mask):
        """Flash Attention.

        Split queries/keys into chunks and do segmented attention
        """
        q = self.q_lin(q)
        k = self.k_lin(k)

        # Split into chunks
        q_chunks = torch.chunk(q, self.n_chunks, dim=1)
        k_chunks = torch.chunk(k, self.n_chunks, dim=1)

        # Calculate attention per chunk
        attn_out = []
        for q_c, k_c in zip(q_chunks, k_chunks):
            attn_weights = torch.matmul(q_c, k_c.transpose(-1, -2))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_out.append(torch.matmul(attn_weights, v_c))

        # Combine chunks
        attn_out = torch.cat(attn_out, dim=1)
        return attn_out


class SparseAttention(Attention):
    """Sparse Attention.

    Apply a mask to the queries before calculating attention.
    """
    def _calculate_attention(self, q, k, v, mask):
        q = self.q_lin(q)
        k = self.k_lin(k)

        # Apply sparse mask to queries
        q = q * mask

        # Attention calculation
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        return attn_out


class GroupedQueryAttention(Attention):
    """Grouped Multi-Head Attention.

    Different heads share the same linear transformation matrices
    on the keys and values. This approach reduces computation costs
    while slightly sacrificing model quality. Models like PaLM and StarCoder
    utilize multi-query attention
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__(d_model, n_heads)
        self.d_k = d_model // n_heads
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        log.info(f'<init>: \n{self}')

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        x = x.transpose(1, 2).contiguous()
        return x

    def _calculate_attention(self, q, k, v, mask):
        q, k, v = self.q_lin(q), self.k_lin(k), self.v_lin(v)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_weights = F.softmax(attn_weights / (self.d_k ** 0.5), dim=-1)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).reshape(attn_out.size(0), -1, self.n_heads * self.d_k)

        output = self.out_lin(attn_out)
        return output
