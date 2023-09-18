"""Tests for the transformer block."""
import pytest
import torch

from small_transformer.base import ModelBase
from small_transformer.model import EncoderDecoderLayer


class MockModule(ModelBase):
    """Mock modules for testing."""
    def __init__(self, **kwargs):
        super().__init__()
        self.called = False

    def forward(self, x, **kwargs):
        """Return the input unchanged."""
        self.called = True
        return x


class MockFFN(MockModule):
    pass


class MockAttn(MockModule):
    def __init__(self, in_features, n_heads):
        super().__init__()


class MockNorm(MockModule):
    def __init__(self, in_features):
        super().__init__()


@pytest.fixture
def inputs():
    """Generate sample inputs for testing"""
    x = torch.rand(16, 32, 512)
    freqs_cos = torch.rand(16, 32, 512)
    freqs_sin = torch.rand(16, 32, 512)
    return x, freqs_cos, freqs_sin


def test_transformations_applied(inputs):
    """Test attention, ffn, and norm modules are called."""
    layer = EncoderDecoderLayer(
        in_features=512,
        n_heads=8,
        ffn=MockFFN(),
        attn_class=MockAttn,
        norm_class=MockNorm
    )
    _ = layer(inputs)
    assert layer.attn.called
    assert layer.ffn_norm.called
    assert layer.attention_norm.called
    assert layer.ffn.called
