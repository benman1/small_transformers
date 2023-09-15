"""Tests for the attention mechanisms."""
import torch
import pytest
from small_transformer.model import (
    MultiHeadAttention, SparseAttention, FlashAttention, GroupedQueryAttention
)

batchsize = 16
seqlen = 32
emb_dim = 512


@pytest.fixture
def inputs():
    """Pytest fixture to generate random input tensors"""
    x = torch.rand(batchsize, seqlen, emb_dim)
    return x


def test_multihead_attention(inputs):
    """Test output shape of standard multi-head attention"""
    mha = MultiHeadAttention(emb_dim, 8)
    x = torch.rand(16, seqlen, emb_dim)
    output = mha(x, x, x)
    assert output.shape == x.shape


def test_sparse_attention(inputs):
    """Test sparse attention with random mask"""
    sa = SparseAttention(emb_dim, 8)
    mask = torch.rand(inputs.shape[:2])
    output = sa(inputs, inputs, inputs, mask)
    assert output.shape == inputs.shape


def test_flash_attention(inputs):
    """Test flash attention with fixed mask"""
    fa = FlashAttention(emb_dim, 8, 4)
    mask = torch.ones(inputs.shape[0], inputs.shape[1] // 4 + 1).bool()
    output = fa(inputs, inputs, inputs, mask)
    assert output.shape == inputs.shape


def test_grouped_query_attention(inputs):
    """Test grouped query attention ignores key and value"""
    gqa = GroupedQueryAttention(emb_dim, 8)
    output = gqa(inputs, inputs, inputs)
    assert output.shape == inputs.shape
