"""Data loader tests."""
import pytest
from small_transformer.utils import TextDataset


@pytest.fixture
def text_file(tmp_path):
    """Temporary file for text data"""
    f = tmp_path / "text.txt"
    f.write_text("This is some sample text data")
    return f


@pytest.fixture
def tokenizer():
    """Mock SentencePiece tokenizer"""

    class MockTokenizer:
        def encode(self, text):
            return [1, 2, 3]

    return MockTokenizer()


def test_text_dataset(text_file, tokenizer):
    """Test dataset loads tokenized encodings"""
    dataset = TextDataset(text_file, tokenizer)

    assert len(dataset) == 1

    sample = dataset[0]
    assert sample == [1, 2, 3]
