"""Dataset loaders."""
from torch.utils.data import Dataset
import sentencepiece as spm


class TextDataset(Dataset):
    """Loads tokenized text data for transformer training"""

    def __init__(self, text_file: str, tokenizer: spm.SentencePieceProcessor):
        """
        Args:
            text_file: path to text file
            tokenizer: sentencepiece tokenizer
        """
        self.text = open(text_file).read()
        self.encodings = tokenizer.encode(self.text)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]
