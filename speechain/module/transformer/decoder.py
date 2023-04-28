"""
    Origin: Sashi Novitasari
    Modification: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch

from speechain.module.abs import Module
from speechain.module.transformer.pos_enc import *
from speechain.module.transformer.attention import *
from speechain.module.transformer.feed_forward import *


class TransformerDecoderLayer(Module):
    """
    A single Transformer decoder layer has:
    · a self multi-head attention sublayer
    · a LayerNorm layer exclusively for the self-attention sublayer
    · a encoder-decoder multi-head attention sublayer
    · a LayerNorm layer exclusively for the encoder-decoder attention sublayer
    · a position-wise feed-forward sublayer
    · a LayerNorm layer exclusively for the feed-forward sublayer
    · a residual dropout layer

    """

    def module_init(self,
                    d_model: int = 512,
                    num_heads: int = 8,
                    att_dropout: float = 0.1,
                    fdfwd_dim: int = 0,
                    fdfwd_activation: str = 'ReLU',
                    fdfwd_dropout: float = 0.1,
                    res_dropout: float = 0.1,
                    layernorm_first: bool = True):
        """
        Represents a single Transformer decoder layer.
        It attends to the source representation and the previous decoder states.

        Args:
            d_model: int
                The dimension of the hidden feature vector in each Transformer layer
            num_heads: int
                The number of attention heads in each Transformer layer
            att_dropout: float
                The dropout rate for the Dropout layer after calculating the weights in each Transformer layer
            fdfwd_dim: int
                The value of the out_features of the first linear feedforward layer and the in_features of the second
                linear feedforward layer in each Transformer layer.
            fdfwd_activation: str
                The name of the activation function of feedforward layers. Should be the name of functions in 'torch.nn'.
            fdfwd_dropout: float
                The dropout rate for the Dropout layer after the first linear feedforward layer in each Transformer layer
            res_dropout: float
                The dropout rate for the Dropout layer before adding the output of each Transformer layer into its input
            layernorm_first: bool
                Whether layernorm is performed before feeding src into sublayers.
                if layernorm_first is True:
                    output = input + Sublayer(LayerNorm(input))
                elif layernorm_first is False:
                    output = LayerNorm(input + Sublayer(input))
        """
        # initialize the self attention layer
        self.self_att = MultiHeadedAttention(num_heads=num_heads, d_model=d_model, dropout=att_dropout)

        # initialize the encoder-decoder attention layer
        self.encdec_att = MultiHeadedAttention(num_heads=num_heads, d_model=d_model, dropout=att_dropout)

        # initialize feedforward layer
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, fdfwd_dim=fdfwd_dim,
                                                    fdfwd_activation=fdfwd_activation, dropout=fdfwd_dropout)

        # initialize layernorm layers
        self.layernorm_first = layernorm_first
        self.self_att_ln = nn.LayerNorm(d_model, eps=1e-6)
        self.encdec_att_ln = nn.LayerNorm(d_model, eps=1e-6)
        self.fdfwd_ln = nn.LayerNorm(d_model, eps=1e-6)

        # initialize residual dropout layer
        self.dropout = nn.Dropout(res_dropout)

    def forward(self,
                tgt: torch.Tensor,
                src: torch.Tensor,
                tgt_mask: torch.Tensor,
                src_mask: torch.Tensor):
        """
        Forward pass of a single Transformer decoder layer.

        Args:
            tgt: (batch, tgt_maxlen, d_model)
                target inputs
            src: (batch, src_maxlen, d_model)
                source representations
            tgt_mask: (batch, tgt_maxlen, tgt_maxlen)
                target mask (so as to not condition on future steps)
            src_mask: (batch, 1, src_maxlen)
                source mask

        Returns:
            The output of this Transformer decoder layer and the attention matrix (self and enc-dec)

        """

        # --- 1. Self Attention Layer part --- #
        # go through the LayerNorm layer before the self attention layer or not
        tgt_norm = self.self_att_ln(tgt) if self.layernorm_first else tgt

        # go through the self attention layer and perform the residual connection
        self_att_hidden, self_attmat = self.self_att(tgt_norm, tgt_norm, tgt_norm, mask=tgt_mask)
        self_att_output = self.dropout(self_att_hidden) + tgt

        # go through the LayerNorm layer after the self attention layer or not
        self_att_output = self.self_att_ln(self_att_output) if not self.layernorm_first else self_att_output

        # --- 2. Enc-Dec Attention Layer part --- #
        # go through the LayerNorm layer before the enc-dec attention layer or not
        self_att_output_norm = self.encdec_att_ln(self_att_output) if self.layernorm_first else self_att_output

        # go through the enc-dec attention layer and perform the residual connection
        encdec_att_hidden, encdec_attmat = self.encdec_att(src, src, self_att_output_norm, mask=src_mask)
        encdec_att_output = self.dropout(encdec_att_hidden) + self_att_output

        # go through the LayerNorm layer after the enc-dec attention layer or not
        encdec_att_output = self.encdec_att_ln(encdec_att_output) if not self.layernorm_first else encdec_att_output

        # --- 3. Positional FeedForward Layer part --- #
        # go through the LayerNorm layer before the feedforward layer or not
        encdec_att_output_norm = self.fdfwd_ln(encdec_att_output) if self.layernorm_first else encdec_att_output

        # go through the feedforward layer and perform the residual connection
        fdfwd_hidden = self.feed_forward(encdec_att_output_norm)
        fdfwd_output = self.dropout(fdfwd_hidden) + encdec_att_output

        # go through the LayerNorm layer after the feedforward layer or not
        fdfwd_output = self.fdfwd_ln(fdfwd_output) if not self.layernorm_first else fdfwd_output

        return fdfwd_output, self_attmat, encdec_attmat


class TransformerDecoder(Module):

    def module_init(self,
                    posenc_type: str = 'mix',
                    posenc_maxlen: int = 5000,
                    posenc_dropout: float = 0.1,
                    posenc_scale: bool = False,
                    posenc_init_alpha: float = 1.0,
                    emb_layernorm: bool = False,
                    emb_scale: bool = True,
                    d_model: int = 512,
                    num_heads: int = 4,
                    num_layers: int = 8,
                    fdfwd_dim: int = 2048,
                    fdfwd_activation: str = 'ReLU',
                    fdfwd_dropout: float = 0.1,
                    att_dropout: float = 0.1,
                    res_dropout: float = 0.1,
                    layernorm_first: bool = True):
        """

        Args:
            posenc_type: str
                Specify the positional encoding type you would like to use in your Transformer blocks.
            posenc_maxlen: int
                Maximal length when calculating the positional encoding.
                Usually, the default value of this argument is enough for the research.
            posenc_dropout: float
                The dropout rate for the Dropout layer after adding the positional encoding to the input
            posenc_scale: bool
                Controls whether the positional encodings are scaled up by a trainable scalar before adding into the
                embedded features or not.
                Reference:
                    'Neural Speech Synthesis with Transformer Network'
                    https://ojs.aaai.org/index.php/AAAI/article/view/4642/4520
            posenc_init_alpha: float = 1.0
                The initial value of the alpha used for positional encoding scaling.
                Only effective when posenc_scale is True.
            emb_layernorm: bool
                Controls whether the embedding vectors are normalized by LayerNorm before adding into the positional
                encoding or not.
            emb_scale: bool
                Controls whether the embedding vectors are scaled up by sqrt(d_model) before adding into the positional
                encoding or not.
            d_model: int
                The dimension of the hidden feature vector in each Transformer layer
            num_heads: int
                The number of attention heads in each Transformer layer
            num_layers: int
                The number of Transformer layers
            att_dropout: float
                The dropout rate for the Dropout layer after calculating the weights in each Transformer layer
            fdfwd_dim: int
                The value of the out_features of the first linear feedforward layer and the in_features of the second
                linear feedforward layer in each Transformer layer.
            fdfwd_activation: str
                The name of the activation function of feedforward layers. Should be the name of functions in 'torch.nn'.
            fdfwd_dropout: float
                The dropout rate for the Dropout layer after the first linear feedforward layer in each Transformer layer
            res_dropout: float
                The dropout rate for the Dropout layer before adding the output of each Transformer layer into its input
            layernorm_first: bool
                controls whether the LayerNorm layer appears at the beginning or at the end of each Transformer layer.
                True means the LayerNorm layer appears at the beginning; False means the LayerNorm layer appears at the end.

        """

        # input_size and output_size initialization
        if self.input_size is not None:
            d_model = self.input_size
        self.output_size = d_model

        # para recording
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.layernorm_first = layernorm_first

        # initialize the positional encoding layer
        self.posenc = PositionalEncoding(posenc_type=posenc_type,
                                         d_model=d_model,
                                         emb_scale=emb_scale,
                                         emb_layernorm=emb_layernorm,
                                         posenc_scale=posenc_scale,
                                         init_alpha=posenc_init_alpha,
                                         max_len=posenc_maxlen,
                                         dropout=posenc_dropout)

        # create num_layers decoder layers and put them in a list
        self.trfm_layers = torch.nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model,
                                    num_heads=num_heads,
                                    att_dropout=att_dropout,
                                    fdfwd_dim=fdfwd_dim,
                                    fdfwd_activation=fdfwd_activation,
                                    fdfwd_dropout=fdfwd_dropout,
                                    res_dropout=res_dropout)
            for _ in range(num_layers)])

        # initialize layernorm layer if necessary
        if self.layernorm_first:
            self.layernorm = nn.LayerNorm(d_model, eps=1e-6)

    @staticmethod
    def subsequent_mask(batch_size, maxlen: int) -> torch.Tensor:
        """
        Mask out subsequent positions (to prevent attending to future positions)
        Transformer helper function.

        Args:
            batch_size:
            maxlen: int
                size of mask (2nd and 3rd dim)

        Returns:

        """
        return ~torch.triu(torch.ones(batch_size, maxlen, maxlen, dtype=torch.bool), diagonal=1)

    def forward(self,
                tgt: torch.Tensor,
                src: torch.Tensor,
                tgt_mask: torch.Tensor,
                src_mask: torch.Tensor,
                return_att: bool = False,
                return_hidden: bool = False):
        """
        Transformer decoder forward pass.

        Args:
            tgt: (batch, tgt_maxlen, d_model)
                embedded targets
            src: (batch, src_maxlen, d_model)
                source representations
            tgt_mask: (batch, 1, tgt_maxlen)
                to mask out target paddings
                Note that a subsequent mask is applied here.
            src_mask: (batch, 1, src_maxlen)
                to mask out source paddings
            return_att:
            return_hidden:

        Returns:
            The output of the Transformer decoder.
            The outputs of each Transformer decoder layer will be returned as a List.
            The attention matrix (self and enc-dec) of each Transformer decoder layer will also be returned as a List.

        """
        assert tgt_mask is not None, "tgt_mask is required for Transformer!"

        # pass the positional encoding layer
        tgt = self.posenc(tgt)

        # generate the diagonal mask for self-attention layers
        batch_size, _, tgt_maxlen = tgt_mask.size()
        tgt_mask = torch.logical_and(tgt_mask.repeat(1, tgt_maxlen, 1),
                                     self.subsequent_mask(batch_size, tgt_maxlen).to(tgt_mask.device))

        # pass the transformer layers
        self_attmat, encdec_attmat, hidden = [], [], []
        for layer in self.trfm_layers:
            tgt, _self_attmat, _encdec_attmat = layer(tgt=tgt, tgt_mask=tgt_mask,
                                                      src=src, src_mask=src_mask)
            self_attmat.append(_self_attmat)
            encdec_attmat.append(_encdec_attmat)
            hidden.append(tgt.clone())

        # pass the layernorm layer if necessary
        if self.layernorm_first:
            tgt = self.layernorm(tgt)

        return tgt, self_attmat, encdec_attmat, hidden
