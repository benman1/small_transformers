"""GPT-like language model.

References:
* One Wide Feedforward is All You Need by Pessoa Pires and others (2023).
"""
import copy
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from small_transformer.base import ModelBase
from small_transformer.model.attention import (
    FlashAttention,
    GroupedQueryAttention,
    SparseAttention,
)
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
    
    TODO: add weight initialization.
    TODO: positional encoding?
    """

    def __init__(
            self,
            vocab_size: int,
            in_features: int,
            n_layers: int,
            n_heads: int,
            d_ff: int,
            attention: str = 'grouped_query',
            norm: str ='layer_norm',
            parameter_sharing: str = 'one_ffn_encodec'
    ):
        """
        Args:
            vocab_size: vocab size of model.
            in_features: dimension of embeddings.
            n_layers: number of encoder/decoder layers.
            n_heads: number of attention heads.
            d_ff: dimensions of feedforward network.
            attention: type of attention mechanism ('grouped_query', 'sparse', 'flash').
            norm: type of normalization layer ('layer_norm', 'rms_norm').
            parameter_sharing: type of parameter sharing for ffn
                ('no', 'one_ffn_encodec').
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.in_features = in_features
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.embed = nn.Embedding(self.vocab_size, self.in_features)
        self.pos_embed = nn.Parameter(torch.randn(1, self.in_features))
        self.attention = attention
        self.norm = norm
        self.parameter_sharing = parameter_sharing

        match self.attention:
            case 'sparse':
                attn_class = SparseAttention
            case 'flash':
                attn_class = FlashAttention
            case _:  # 'grouped_query':
                attn_class = GroupedQueryAttention

        match self.norm:
            case 'layer_norm':
                norm_class = LayerNorm
            case _:  # rms_norm
                norm_class = RMSNorm

        match self.parameter_sharing:
            case 'one_ffn_encodec':
                ffn = FeedForwardNetwork(self.in_features, self.d_ff)
                self.encoder = nn.ModuleList([
                    EncoderDecoderLayer(
                        self.in_features,
                        self.n_heads,
                        ffn,
                        attn_class,
                        norm_class
                    )
                    for _ in range(self.n_layers)
                ])
            case _:
                encoder_layer = EncoderDecoderLayer(
                    self.in_features,
                    self.n_heads,
                    FeedForwardNetwork(self.in_features, self.d_ff),
                    attn_class,
                    norm_class
                )
                self.encoder = nn.ModuleList([copy.deepcopy(encoder_layer)
                                              for _ in range(self.n_layers)])

        self.lm_head = nn.Linear(self.in_features, self.vocab_size)

    def forward(self, x):
        x = self.embed(x) + self.pos_embed

        for encoder in self.encoder:
            x = encoder(x)

        logits = self.lm_head(x)
        return logits


class EncoderDecoderLayer(ModelBase):
    """Transformer encoder-decoder layer."""

    def __init__(
            self,
            in_features: int,
            n_heads: int,
            ffn,
            attn_class,
            norm_class
    ):
        super().__init__()
        """Init.
        
        Args:
            in_features: dimensions of model embeddings
            n_heads: number of attention heads
            ffn: feedforward network with in_features dimensions and
                arbitrary out_features.
            attn_class: attention class to use 
            norm_class: normalization class to use
        """
        self.attn = attn_class(in_features, n_heads)
        self.norm1 = norm_class(in_features)
        self.ffn = ffn

        self.norm2 = norm_class(in_features)

        log.info(f'<init>: \n{self}')

    def forward(self, x):
        """Forward pass."""
        attn_out = self.attn.forward(x, x, x)
        out1 = attn_out + x

        ffn_out = self.ffn.forward(out1)
        out2 = ffn_out + out1
        return out2


class FeedForwardNetwork(ModelBase):
    def __init__(self, in_features, out_features, act_fun: Callable = F.gelu):
        """Init.

        Use GLU gating by default.
        Reference: GLU Variants Improve Transformer, Noam Shazeer (2020).
        """
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, in_features)
        self.act_fun = act_fun
        log.info(f'<init>: \n{self}')

    def forward(self, x):
        """
        Use GLU gating by default.
        """
        x = self.linear1(x)
        x = self.act_fun(x)
        x = self.linear2(x)
        return x
