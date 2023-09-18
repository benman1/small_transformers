"""GPT-like language model.

References:
* One Wide Feedforward is All You Need by Pessoa Pires and others (2023).
"""
import copy
import math

import torch.nn as nn
from torch import Tensor

from small_transformer.base import ModelBase
from small_transformer.model.attention import AttentionTypes, get_attention_class
from small_transformer.model.encoder_decoder import EncoderDecoderLayer
from small_transformer.model.feedforward import FeedForwardNetwork
from small_transformer.model.normalization import LayerNorm, RMSNorm
from small_transformer.utils import setup_logger

log = setup_logger(__name__)


class GPTS(ModelBase):
    """GPTS - a small GPT.

    This implements the core transformer architecture

    The EncoderDecoderLayer contains the multi-head self attention and
    feedforward network, and is shared between the encoder and decoder.
    The attention is implemented to use grouped queries, matching a GPT-style
    decoder-only architecture.
    """

    def __init__(
            self,
            vocab_size: int,
            in_features: int,
            n_layers: int,
            n_heads: int,
            attention: AttentionTypes = "Flash",
            norm: str = 'layer_norm',
            parameter_sharing: str = 'one_ffn_encodec'
    ):
        """
        Args:
            vocab_size: vocab size of model.
            in_features: dimension of embeddings.
            n_layers: number of encoder/decoder layers.
            n_heads: number of attention heads.
            d_ff: dimensions of feedforward network.
            attention: type of attention mechanism (e.g. 'Flash').
            norm: type of normalization layer ('layer_norm', 'rms_norm').
            parameter_sharing: type of parameter sharing for ffn
                ('no', 'one_ffn_encodec').
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.in_features = in_features
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = 0.1
        self.embedding = nn.Embedding(self.vocab_size, self.in_features)
        self.dropout = nn.Dropout(self.dropout)
        self.attn_class = get_attention_class(attention_type=attention)
        self.norm = norm
        self.parameter_sharing = parameter_sharing

        match self.norm:
            case 'layer_norm':
                self.norm_class = LayerNorm
            case _:  # rms_norm
                self.norm_class = RMSNorm

        match self.parameter_sharing:
            case 'one_ffn_encodec':
                ffn = FeedForwardNetwork(self.in_features, self.in_features)
                self.encoders = nn.ModuleList([
                    EncoderDecoderLayer(
                        self.in_features,
                        self.n_heads,
                        ffn,
                        self.attn_class,
                        self.norm_class
                    )
                    for _ in range(self.n_layers)
                ])
            case _:
                encoder_layer = EncoderDecoderLayer(
                    self.in_features,
                    self.n_heads,
                    FeedForwardNetwork(self.in_features, self.in_features),
                    self.attn_class,
                    self.norm_class
                )
                self.encoders = nn.ModuleList([copy.deepcopy(encoder_layer)
                                               for _ in range(self.n_layers)])

        self.linear = nn.Linear(self.in_features, self.vocab_size)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """Propagate values through the network.

        Args:
            x: Tensor, shape ``[seq_len, batch_size]``
            mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.dropout(
            self.embedding(x) * math.sqrt(self.in_features)
        )
        src = self.dropout(src)
        for encoder in self.encoders:
            src = encoder(src)

        logits = self.linear(src)
        return logits
