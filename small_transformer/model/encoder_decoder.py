"""Encoder decoder layer implementation."""
from small_transformer.base import ModelBase
from small_transformer.model import log


class EncoderDecoderLayer(ModelBase):
    """Transformer encoder-decoder layer block."""

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
        self.attention_norm = norm_class(in_features)
        self.ffn_norm = norm_class(in_features)
        self.ffn = ffn

        log.info(f'<init>: \n{self}')

    def forward(self, x):
        """Forward pass."""
        attn_out = self.attn.forward(self.attention_norm(x))
        out1 = attn_out + x

        ffn_out = self.ffn.forward(self.ffn_norm(out1))
        out2 = ffn_out + out1
        return out2
