"""Test normalizations."""
import pytest
import torch

from small_transformer.model import LayerNorm, RMSNorm


@pytest.fixture
def random_inputs():
    """Generate random float tensor as test input"""
    return torch.rand(16, 32, 64)


def test_layer_norm(random_inputs):
    """Test output shape/value range for LayerNorm layer"""
    ln = LayerNorm(random_inputs.size(-1))
    out = ln(random_inputs)

    assert out.shape == random_inputs.shape


def test_layer_norm_grads(random_inputs):
    """Test LayerNorm gradients for scale/shift parameters"""
    ln = LayerNorm(random_inputs.size(-1))
    out = ln(random_inputs)

    out.mean().backward()

    assert ln.weight.grad is not None
    assert ln.bias.grad is not None


def test_rms_norm(random_inputs):
    """Test output shape/value range for RMSNorm layer"""
    rn = RMSNorm(random_inputs.size(-1))
    out = rn(random_inputs)

    assert out.shape == random_inputs.shape
    assert out.min() >= -1
    assert out.max() <= 1
