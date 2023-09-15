"""Tests for the tokenizer."""
import os

import pytest

from small_transformer.model import load_tokenizer, train_tokenizer


@pytest.fixture
def text_file(tmp_path):
    """Temporary file for text data"""
    f = tmp_path / "text.txt"
    f.write_text("This is some sample text data")
    return f


def test_train_tokenizer(text_file):
    """Test training."""
    train_tokenizer(modelfile="test_tokenizer", vocab_size=273, textfile=str(text_file))
    assert os.path.exists("test_tokenizer.model")


def test_load_tokenizer(text_file):
    """Test loading and applying the tokenizer."""
    tokenizer = load_tokenizer(modelfile="resources/test_tokenizer")
    encoding = tokenizer.encode("This is a test")
    assert tokenizer.decode(encoding) == "This is a test"
