"""Dataset loaders."""
import json

import torch
from torch.utils.data import Dataset


class PretokenizedDataLoader:
    """Loads tokenized text data for transformer training.

    Includes functionality for saving and loading a tokenized dataset.
    Iterates through torchtext dataset for padding and batching
    """

    def __init__(
            self,
            dataset_path: str,
            batch_size: int = 32,
            max_seq_len: int = 128
    ):
        """Init.
        Args:
          dataset_path: path to tokenized torchtext dataset.
          batch_size (int): batch size.
          max_seq_len (int): maximum sequence length.
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def pretokenize(self, tokenizer, dataset: Dataset):
        """Saves tokenized torchtext dataset to JSON lines file.

        Args:
          tokenizer: a tokenizer, for example, a sentencepiece BPE.
          dataset: a torchtext dataset to tokenize.
        """
        with open(self.dataset_path, 'w') as f:
            for example in dataset:
                # TODO: add tokenization here!
                json.dump(example, f)
                f.write('\n')

    def __len__(self):
        """Number of lines in the file.

        TODO: check this corresponds to the number of entries!
        """
        return sum(1 for _ in open(self.dataset_path))

    def __iter__(self):
        """Returns iterator over dataset.

        Padding batches and truncating to max length.
        No shuffling, since reading from a sequential input (file).
        """
        """Yields padded batches while iterating over dataset"""
        batch = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                tokens = json.loads(line)
                batch.append(tokens)
                if len(batch) == self.batch_size:
                    padded = pad_truncate_batch(batch, self.max_seq_len)
                    yield torch.tensor(padded)
                    batch = []

            if batch:
                padded = pad_truncate_batch(batch, self.max_seq_len)
                yield torch.tensor(padded)


def pad_truncate_batch(batch, max_len: int):
    """Pad and truncate a sequence batch.

    Pads sequences to max length, truncating if longer.
    """
    padded = []
    for sequence in batch:
        if len(sequence) > max_len:
             padded.append(sequence[:max_len])
        else:
            padding = [0] * (max_len - len(sequence))
            padded.append(sequence + padding)
        padded.append(sequence)
    return padded
