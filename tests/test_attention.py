"""Tests for the attention mechanisms."""
import pytest
import torch

from small_transformer.model import Attention, FlashAttention, MultiHeadAttention

batchsize = 16
seqlen = 32
emb_dim = 512


@pytest.fixture
def inputs():
    """Pytest fixture to generate random input tensors"""
    x = torch.rand(batchsize, seqlen, emb_dim)
    return x


def test_attention(inputs):
    """Test output shape of softmax attention"""
    attn = Attention(emb_dim, 8)
    x = torch.rand(16, seqlen, emb_dim)
    output = attn(x)
    assert output.shape == x.shape


def test_mha_attention(inputs):
    """Test output shape of multi-head attention"""
    mha = MultiHeadAttention(emb_dim, 8)
    x = torch.rand(16, seqlen, emb_dim)
    output = mha(x)
    assert output.shape == x.shape


def test_flash_attention(inputs):
    """Test output shape of standard multi-head attention"""
    fa = FlashAttention(emb_dim, 8)
    x = torch.rand(16, seqlen, emb_dim)
    output = fa(x)
    assert output.shape == x.shape
