from typing import Dict, Any

import torch
from torch import nn

from speechain.module.abs import Module
from speechain.module.conformer.pos_enc import RelPositionalEncoding
from speechain.module.conformer.attention import RelPosMultiHeadedAttention
from speechain.module.transformer.feed_forward import PositionwiseFeedForward
from speechain.module.prenet.conv1d import Conv1dEv

from speechain.utilbox.train_util import swish_activation

class ConvolutionModule(Module):

    def module_init(self, input_channels: int = None, depthwise_kernel_size: int = 31):

        if input_channels is None:
            assert self.input_size is not None
            input_channels = self.input_size

        self.pointwise_conv1 = Conv1dEv(
            in_channels=input_channels, out_channels=2 * input_channels,
            kernel_size=1, stride=1, padding_mode='valid', bias=True
        )

        self.depthwise_conv = Conv1dEv(
            in_channels=input_channels, out_channels=input_channels,
            kernel_size=depthwise_kernel_size, stride=1, padding_mode='same', groups=input_channels
        )

        self.batch_norm = nn.BatchNorm1d(input_channels)

        self.pointwise_conv2 = Conv1dEv(
            in_channels=input_channels, out_channels=input_channels,
            kernel_size=1, stride=1, padding_mode='valid', bias=True
        )

    def forward(self, feat: torch.Tensor):

        # exchange the temporal dimension and the feature dimension
        feat = feat.transpose(1, 2)

        # GLU mechanism
        feat = self.pointwise_conv1(feat)  # (batch, 2*channel, dim)
        feat = nn.functional.glu(feat, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        feat = self.depthwise_conv(feat)
        feat = swish_activation(self.batch_norm(feat))

        feat = self.pointwise_conv2(feat)
        return feat.transpose(1, 2)


class ConformerEncoderLayer(Module):

    def module_init(self,
                    d_model: int = 512,
                    num_heads: int = 8,
                    att_dropout: float = 0.1,
                    depthwise_kernel_size: int = 31,
                    fdfwd_dim: int = 2048,
                    fdfwd_type: str = 'linear',
                    fdfwd_activation: str = 'ReLU',
                    fdfwd_args: Dict[str, Any] = {},
                    fdfwd_dropout: float = 0.1,
                    res_dropout: float = 0.1,
                    layernorm_first: bool = True):

        # initialize feadforward layer in front of MHA and Conv modules
        self.front_feed_forward = PositionwiseFeedForward(d_model=d_model, fdfwd_dim=fdfwd_dim, fdfwd_type=fdfwd_type,
                                                          fdfwd_activation=fdfwd_activation, fdfwd_args=fdfwd_args,
                                                          dropout=fdfwd_dropout)
        self.front_fdfwd_layernorm = nn.LayerNorm(d_model, eps=1e-6)

        #
        self.relpos_mha = RelPosMultiHeadedAttention(d_model=d_model, num_heads=num_heads, dropout=att_dropout)
        self.mha_layernorm = nn.LayerNorm(d_model, eps=1e-6)

        #
        self.conv_module = ConvolutionModule(input_size=d_model, depthwise_kernel_size=depthwise_kernel_size)
        self.conv_layernorm = nn.LayerNorm(d_model, eps=1e-6)

        # initialize feadforward layer behind MHA and Conv modules
        self.rear_feed_forward = PositionwiseFeedForward(d_model=d_model, fdfwd_dim=fdfwd_dim, fdfwd_type=fdfwd_type,
                                                         fdfwd_activation=fdfwd_activation, fdfwd_args=fdfwd_args,
                                                         dropout=fdfwd_dropout)
        self.rear_fdfwd_layernorm = nn.LayerNorm(d_model, eps=1e-6)

        # initialize layernorm layers, each sublayer has an exclusive LayerNorm layer
        self.layernorm_first = layernorm_first

        # initialize residual dropout layer
        self.dropout = nn.Dropout(res_dropout)

    def forward(self, src: torch.Tensor, mask: torch.Tensor, posenc: torch.Tensor):
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
        'Front Positional FeedForward Layer part'
        # go through the LayerNorm layer before the feedforward layer or not
        src_norm = self.front_fdfwd_layernorm(src) if self.layernorm_first else src

        # go through the feedforward layer and perform the residual connection
        front_fdfwd_hidden = self.front_feed_forward(src_norm)
        front_fdfwd_output = 0.5 * self.dropout(front_fdfwd_hidden) + src

        # go through the LayerNorm layer after the feedforward layer or not
        front_fdfwd_output = self.front_fdfwd_layernorm(front_fdfwd_output) if not self.layernorm_first else front_fdfwd_output

        'Relative Positional Multi-head Attention Layer part'
        # go through the LayerNorm layer before the multi-head attention layer or not
        front_fdfwd_output_norm = self.mha_layernorm(front_fdfwd_output) if self.layernorm_first else front_fdfwd_output

        # go through the multi-head attention layer and perform the residual connection
        relpos_mha_hidden, attmat = self.relpos_mha(front_fdfwd_output_norm, front_fdfwd_output_norm,
                                                    front_fdfwd_output_norm, mask, posenc)
        relpos_mha_output = self.dropout(relpos_mha_hidden) + front_fdfwd_output

        # go through the LayerNorm layer after the multi-head attention layer or not
        relpos_mha_output = self.mha_layernorm(relpos_mha_output) if not self.layernorm_first else relpos_mha_output

        'Convolutional Module part'
        # go through the LayerNorm layer before the feedforward layer or not
        relpos_mha_output_norm = self.conv_layernorm(relpos_mha_output) if self.layernorm_first else relpos_mha_output

        # go through the feedforward layer and perform the residual connection
        conv_hidden = self.conv_module(relpos_mha_output_norm)
        conv_output = self.dropout(conv_hidden) + relpos_mha_output

        # go through the LayerNorm layer after the feedforward layer or not
        conv_output = self.conv_layernorm(conv_output) if not self.layernorm_first else conv_output

        'Rear Positional FeedForward Layer part'
        # go through the LayerNorm layer before the feedforward layer or not
        conv_output_norm = self.rear_fdfwd_layernorm(conv_output) if self.layernorm_first else conv_output

        # go through the feedforward layer and perform the residual connection
        rear_fdfwd_hidden = self.rear_feed_forward(conv_output_norm)
        rear_fdfwd_output = 0.5 * self.dropout(rear_fdfwd_hidden) + conv_output

        # go through the LayerNorm layer after the feedforward layer or not
        rear_fdfwd_output = self.rear_fdfwd_layernorm(rear_fdfwd_output) if not self.layernorm_first else rear_fdfwd_output

        return rear_fdfwd_output, attmat

class ConformerEncoder(Module):
    """

    """

    def module_init(self,
                    posenc_type: str = 'mix',
                    posenc_maxlen: int = 5000,
                    posenc_dropout: float = 0.1,
                    emb_scale: bool = False,
                    d_model: int = 512,
                    num_heads: int = 4,
                    num_layers: int = 8,
                    att_dropout: float = 0.1,
                    depthwise_kernel_size: int = 31,
                    fdfwd_dim: int = 2048,
                    fdfwd_type: str = 'linear',
                    fdfwd_activation: str = 'ReLU',
                    fdfwd_args: Dict[str, Any] = {},
                    fdfwd_dropout: float = 0.1,
                    res_dropout: float = 0.1,
                    layernorm_first: bool = True,
                    uni_direction: bool = False):

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

        self.posenc = RelPositionalEncoding(posenc_type=posenc_type,
                                            d_model=d_model,
                                            emb_scale=emb_scale,
                                            max_len=posenc_maxlen,
                                            dropout=posenc_dropout)

        # initialize transformer layers
        self.cfm_layers = torch.nn.ModuleList([
            ConformerEncoderLayer(d_model=d_model,
                                  num_heads=num_heads,
                                  att_dropout=att_dropout,
                                  depthwise_kernel_size=depthwise_kernel_size,
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
        # add position encoding to word embeddings
        src, posenc = self.posenc(src)

        # generate the low-triangular mask for self-attention layers
        if self.uni_direction:
            batch_size, _, src_maxlen = mask.size()
            mask = torch.logical_and(mask.repeat(1, src_maxlen, 1),
                                     self.subsequent_mask(batch_size, src_maxlen).to(mask.device))

        # go through the Conformer layers
        attmat, hidden = [], []
        for l in range(len(self.cfm_layers)):
            src, _tmp_attmat = self.cfm_layers[l](src, mask, posenc)
            attmat.append(_tmp_attmat)
            hidden.append(src.clone())

        # go through the final layernorm layer if necessary
        if self.layernorm_first:
            src = self.layernorm(src)

        return src, mask, attmat, hidden