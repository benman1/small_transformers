"""Tests for the tokenizer.

TODO: check this properly!
"""
import pytest

from small_transformer.model import train_tokenizer, load_tokenizer


@pytest.fixture
def tokenizer():
    """Load a tokenizer"""
    return load_tokenizer("test_tokenizer.model")


def test_train_tokenizer(tokenizer):
    encoding = tokenizer.encode("This is a test")
    assert tokenizer.decode(encoding) == "This is a test"
