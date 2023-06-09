"""
    Origin: Sashi Novitasari
    Modification: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch
from torch import nn
from typing import List, Dict, Any

from speechain.module.abs import Module
from speechain.module.transformer.pos_enc import PositionalEncoding
from speechain.module.transformer.attention import MultiHeadedAttention
from speechain.module.transformer.feed_forward import PositionwiseFeedForward


class TransformerEncoderLayer(Module):
    """
    A single Transformer encoder layer has:
    · a Multi-head attention sublayer
    · a LayerNorm layer exclusively for the attention sublayer
    · a position-wise feed-forward sublayer
    · a LayerNorm layer exclusively for the feed-forward sublayer
    · a residual dropout layer

    """

    def module_init(self,
                    d_model: int = 512,
                    num_heads: int = 8,
                    scale_dp_by_head: bool = False,
                    att_dropout: float = 0.1,
                    fdfwd_dim: int = 2048,
                    fdfwd_type: str = 'linear',
                    fdfwd_activation: str = 'ReLU',
                    fdfwd_args: Dict[str, Any] = {},
                    fdfwd_dropout: float = 0.1,
                    res_dropout: float = 0.1,
                    layernorm_first: bool = True):
        """

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
            fdfwd_type: str
                The type of the feed-forward layer. 'linear' means the Linear layer while 'conv' means the Conv1d layer.
            fdfwd_activation: str
                The name of the activation function of feedforward layers. Should be the name of functions in 'torch.nn'.
            fdfwd_kernel: int
                The kernal size of the Conv1d feed-forward layer. This argument is not effective if fdfwd_type == 'linear'.
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
        # initialize multi-head attention layer
        self.multihead_att = MultiHeadedAttention(d_model=d_model, num_heads=num_heads, dropout=att_dropout,
                                                  scale_dp_by_head=scale_dp_by_head)

        # initialize feedforward layer
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, fdfwd_dim=fdfwd_dim, fdfwd_type=fdfwd_type,
                                                    fdfwd_activation=fdfwd_activation, fdfwd_args=fdfwd_args,
                                                    dropout=fdfwd_dropout)

        # initialize residual dropout layer
        self.dropout = nn.Dropout(res_dropout)

        # initialize layernorm layers, each sublayer has an exclusive LayerNorm layer
        self.layernorm_first = layernorm_first
        self.att_layernorm = nn.LayerNorm(d_model, eps=1e-6)
        self.fdfwd_layernorm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src: torch.Tensor, mask: torch.Tensor):
        """
        Forward pass for a single transformer encoder layer.

        Args:
            src: (batch, src_maxlen, d_model)
                source input for the encoder
            mask: (batch, 1, src_maxlen)
                input mask

        Returns:
            The output of this Transformer encoder layer and the attention matrix

        """

        'Multi-head Attention Layer part'
        # go through the LayerNorm layer before the multi-head attention layer or not
        src_norm = self.att_layernorm(src) if self.layernorm_first else src

        # go through the multi-head attention layer and perform the residual connection
        att_hidden, attmat = self.multihead_att(src_norm, src_norm, src_norm, mask)
        att_output = self.dropout(att_hidden) + src

        # go through the LayerNorm layer after the multi-head attention layer or not
        att_output = self.att_layernorm(att_output) if not self.layernorm_first else att_output

        'Positional FeedForward Layer part'
        # go through the LayerNorm layer before the feedforward layer or not
        att_output_norm = self.fdfwd_layernorm(att_output) if self.layernorm_first else att_output

        # go through the feedforward layer and perform the residual connection
        fdfwd_hidden = self.feed_forward(att_output_norm)
        fdfwd_output = self.dropout(fdfwd_hidden) + att_output

        # go through the LayerNorm layer after the feedforward layer or not
        fdfwd_output = self.fdfwd_layernorm(fdfwd_output) if not self.layernorm_first else fdfwd_output

        return fdfwd_output, attmat


class TransformerEncoder(Module):
    """
        The Transformer encoder for any Sequence-to-Sequence tasks.
        Reference:
            Attention is all you need
            https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

        Our Transformer encoder implements the following properties:
            1. Different positional encoding. (Mix or Sep)
            2. Different positions of the LayerNorm layer (first or last)
            3. Time Frame Downsampling (pool or concat)
        For the details, please refer to the docstrings of PositionalEncoding and TransformerEncoderLayer.

        In our Transformer implementation, there are 4 places to place the Dropout layers:
            1. After adding the positional encoding into the embedded features.
            2. After the softmax operation and before reweighting all the values by these weights in the
                multi-head attention layer.
            3. Between two feedforward linear layers there will be a Dropout layer.
            4. Before performing residual connect in a Transformer layer.

    """

    def module_init(self,
                    posenc_type: str = 'mix',
                    posenc_maxlen: int = 5000,
                    posenc_dropout: float = 0.1,
                    posenc_scale: bool = False,
                    posenc_init_alpha: float = 1.0,
                    emb_layernorm: bool = False,
                    emb_scale: bool = False,
                    d_model: int = 512,
                    num_heads: int = 4,
                    num_layers: int = 8,
                    scale_dp_by_head: bool = False,
                    att_dropout: float = 0.1,
                    fdfwd_dim: int = 2048,
                    fdfwd_type: str = 'linear',
                    fdfwd_activation: str = 'ReLU',
                    fdfwd_args: Dict[str, Any] = {},
                    fdfwd_dropout: float = 0.1,
                    res_dropout: float = 0.1,
                    layernorm_first: bool = True,
                    uni_direction: bool = False):
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
            fdfwd_type: str
                The type of the feed-forward layer. 'linear' means the Linear layer while 'conv' means the Conv1d layer.
            fdfwd_activation: str
                The name of the activation function of feedforward layers. Should be the name of functions in 'torch.nn'.
            fdfwd_dropout: float
                The dropout rate for the Dropout layer after the first linear feedforward layer in each Transformer layer
            res_dropout: float
                The dropout rate for the Dropout layer before adding the output of each Transformer layer into its input
            uni_direction: bool = False
                Whether the encoder is unidirectional or not. If True, the attention matrix will be masked into a
                lower-triangular matrix.
            layernorm_first: bool
                controls whether the LayerNorm layer appears at the beginning or at the end of each Transformer layer.
                    True means the LayerNorm layer appears at the beginning
                    False means the LayerNorm layer appears at the end.
                For LayerNorm first, there will be an additional LayerNorm at the end of the Transformer Encoder to
                perform the final normalization.

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
        self.uni_direction = uni_direction

        # initialize positional encoding layer
        self.posenc = PositionalEncoding(posenc_type=posenc_type,
                                         d_model=d_model,
                                         emb_scale=emb_scale,
                                         emb_layernorm=emb_layernorm,
                                         posenc_scale=posenc_scale,
                                         init_alpha=posenc_init_alpha,
                                         max_len=posenc_maxlen,
                                         dropout=posenc_dropout)

        # initialize transformer layers
        self.trfm_layers = torch.nn.ModuleList([
            TransformerEncoderLayer(d_model=d_model,
                                    num_heads=num_heads,
                                    scale_dp_by_head=scale_dp_by_head,
                                    att_dropout=att_dropout,
                                    fdfwd_dim=fdfwd_dim,
                                    fdfwd_type=fdfwd_type,
                                    fdfwd_activation=fdfwd_activation,
                                    fdfwd_args=fdfwd_args,
                                    fdfwd_dropout=fdfwd_dropout,
                                    res_dropout=res_dropout,
                                    layernorm_first=layernorm_first)
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

    def forward(self, src: torch.Tensor, mask: torch.Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        Args:
            src: (batch_size, src_maxlen, embed_size)
                embedded src inputs,
            mask: (batch_size, 1, src_maxlen)
                indicates padding areas (zeros where padding)

        Returns:
            The output of the Transformer encoder with its mask.
            The outputs of each Transformer encoder layer will be returned as a List.
            The attention matrix of each Transformer encoder layer will also be returned as a List.

        """

        # add position encoding to word embeddings
        src = self.posenc(src)

        # generate the low-triangular mask for self-attention layers
        if self.uni_direction:
            batch_size, _, src_maxlen = mask.size()
            mask = torch.logical_and(mask.repeat(1, src_maxlen, 1),
                                     self.subsequent_mask(batch_size, src_maxlen).to(mask.device))

        # go through the Transformer layers
        attmat, hidden = [], []
        for l in range(len(self.trfm_layers)):
            src, _tmp_attmat = self.trfm_layers[l](src, mask)
            attmat.append(_tmp_attmat)
            hidden.append(src.clone())

        # go through the final layernorm layer if necessary
        if self.layernorm_first:
            src = self.layernorm(src)

        return src, mask, attmat, hidden
