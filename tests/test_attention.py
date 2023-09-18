"""Tests for the attention mechanisms."""
import pytest
import torch

from small_transformer.model import Attention

batchsize = 16
seqlen = 32
emb_dim = 512


@pytest.fixture
def inputs():
    """Pytest fixture to generate random input tensors"""
    x = torch.rand(batchsize, seqlen, emb_dim)
    return x


def test_attention(inputs):
    """Test output shape of standard multi-head attention"""
    attn = Attention(emb_dim, 8)
    x = torch.rand(16, seqlen, emb_dim)
    output = attn(x)
    assert output.shape == x.shape


def test_multihead_attention(inputs):
    """Test output shape of standard multi-head attention"""
    mha = Attention(emb_dim, 8, multi_head=True)
    x = torch.rand(16, seqlen, emb_dim)
    output = mha(x)
    assert output.shape == x.shape
