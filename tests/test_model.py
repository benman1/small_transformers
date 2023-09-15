"""Tests for the model class."""
import pytest
import torch

from small_transformer.model import GPTS


vocab_size = 1000
d_model = 512
n_layers = 6
n_heads = 8
d_ff = 2048


@pytest.fixture
def model():
    return GPTS(vocab_size, d_model, n_layers, n_heads, d_ff)


def test_forward(model):
    x = torch.randint(low=0, high=vocab_size, size=(16, 32))
    out = model(x)
    assert out.shape == (16, 32, model.vocab_size)


def test_multiple_layers(model):
    assert len(model.encoder) == model.n_layers


def test_attention_heads(model):
    attn = model.encoder[0].attn
    assert attn.n_heads == model.n_heads


def test_feedforward_dim(model):
    for encoder in model.encoder:
        assert encoder.ffn.linear1.out_features == model.d_ff
    for encoder in model.encoder:
        assert encoder.ffn.linear2.in_features == model.d_ff
