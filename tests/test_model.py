"""Tests for the model class."""
import pytest
import torch

from small_transformer.model import GPTS

vocab_size = 1000
d_model = 512
n_layers = 6
n_heads = 8
n_batches = 16


@pytest.fixture
def model():
    """Instantiate model for reuse."""
    return GPTS(vocab_size, d_model, n_layers, n_heads)


def test_forward(model):
    """Test a forward pass."""
    x = torch.randint(low=0, high=vocab_size, size=(n_batches, 32))
    out = model(x)
    assert out.shape == (16, 1, model.vocab_size)


def test_multiple_layers(model):
    """Test layer initialization."""
    assert len(model.encoders) == model.n_layers


def test_attention_heads(model):
    """Test attention heads match."""
    attn = model.encoders[0].attn
    assert attn.n_heads == model.n_heads


def test_feedforward_dim(model):
    """Test input dimensions in feedforward network."""
    for encoder in model.encoders:
        assert encoder.ffn.linear1.out_features == model.d_ff
    for encoder in model.encoders:
        assert encoder.ffn.linear2.in_features == model.d_ff
