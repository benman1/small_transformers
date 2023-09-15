"""Tokenizer."""
import os

import sentencepiece as spm


def train_tokenizer(
        textfile: str = "train.txt",
        vocab_size: int = 32000,
        modelfile: str = "tokenizer"
):
    """Train and save a sentencepiece tokenizer model."""
    spm.SentencePieceTrainer.train(
        input=textfile,
        model_prefix=modelfile,
        vocab_size=vocab_size,
        character_coverage=1.0,
        split_digits=True,
        model_type='bpe',
        num_threads=os.cpu_count(),
        input_format = "text",
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity"
    )


def load_tokenizer(modelfile: str = "tokenizer") -> spm.SentencePieceProcessor:
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f"{modelfile}.model")
    return tokenizer
