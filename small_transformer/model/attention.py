"""Attention mechanisms.

Attention mechanisms are essential components of Transformers,
and full attention, as well as sparse attention, can be employed.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from small_transformer.base import ModelBase
from small_transformer.utils import setup_logger


log = setup_logger(__name__)


class MultiHeadAttention(ModelBase):

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        log.info(f'<init>: \n{self}')

    def forward(self, query, key, value):
        # QKV projections
        qkv = self.qkv_proj(query)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Head split
        q = q.reshape(q.shape[0], q.shape[1], self.n_heads, self.d_k)
        k = k.reshape(k.shape[0], k.shape[1], self.n_heads, self.d_k)
        v = v.reshape(v.shape[0], v.shape[1], self.n_heads, self.d_k)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_k ** 0.5
        scores = F.softmax(scores, dim=-1)
        context = torch.matmul(scores, v)

        # Head concat and proj
        context = context.transpose(1, 2).reshape(
            context.shape[0], context.shape[1], self.d_model
        )
        output = self.out_proj(context)

        return output


class SparseAttention(MultiHeadAttention):

    def __init__(self, d_model, n_heads):
        super().__init__(d_model, n_heads)
        log.info(f'<init>: \n{self}')

    def forward(self, query, key, value, mask=None):
        # Apply mask to queries
        query = query * mask[:, None, :]

        # Attention
        context = super().forward(query, key, value)
        return context


class FlashAttention(MultiHeadAttention):

    def __init__(self, d_model, n_heads, n_chunks):
        super().__init__(d_model, n_heads)
        self.n_chunks = n_chunks
        log.info(f'<init>: \n{self}')

    def forward(self, query, key, value, mask=None):
        # Segment into chunks
        q_chunks = torch.chunk(query, self.n_chunks, dim=-2)
        k_chunks = torch.chunk(key, self.n_chunks, dim=-2)
        v_chunks = torch.chunk(value, self.n_chunks, dim=-2)

        # Attention on chunks
        context = []
        for q, k, v in zip(q_chunks, k_chunks, v_chunks):
            c = super().forward(q, k, v)
            context.append(c)

        # Concat chunks
        context = torch.cat(context, dim=-2)
        return context


class GroupedQueryAttention(MultiHeadAttention):
    """Grouped Multi-Head Attention.

    Different heads share the same linear transformation matrices
    on the keys and values. This approach reduces computation costs
    while slightly sacrificing model quality. Models like PaLM and StarCoder
    utilize multi-query attention

    The key difference from standard MHA is that q, k, v all come from the same x input.
    """

    def __init__(self, d_model, n_heads):
        super().__init__(d_model, n_heads)
        log.info(f'<init>: \n{self}')

    def forward(self, query, key, value, mask=None):
        qkv = self.qkv_proj(query)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Attention
        q = q.reshape(q.shape[0], q.shape[1], self.n_heads, self.d_k)
        k = k.reshape(k.shape[0], k.shape[1], self.n_heads, self.d_k)
        v = v.reshape(v.shape[0], v.shape[1], self.n_heads, self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_k ** 0.5
        scores = F.softmax(scores, dim=-1)
        context = torch.matmul(scores, v)

        # Output proj
        context = context.transpose(1, 2).reshape(
            context.shape[0], context.shape[1], self.d_model
        )
        output = self.out_proj(context)

        return output
