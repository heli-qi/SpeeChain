"""
    Origin: Sashi Novitasari
    Modification: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import math
from typing import Dict

import torch

from speechain.module.abs import Module


class PositionalEncoding(Module):
    """
    Pre-compute position encodings (PE). In forward pass, this module adds the positional encodings to the embedded
    feature vectors to make the Transformer aware of the positional information of the sequences.

    """

    def module_init(self,
                    posenc_type: str = 'mix',
                    d_model: int = 512,
                    emb_scale: bool = False,
                    emb_layernorm: bool = False,
                    posenc_scale: bool = False,
                    init_alpha: float = 1.0,
                    max_len: int = 5000,
                    dropout: float = 0.0):
        """
        Positional Encoding with maximum length max_len.

        Args:
            posenc_type: str
                The type of positional encoding (must be either 'mix' or 'sep').
                For the 'mix' type, sin is applied to the odd dimensions and cos is applied to the even dimensions.
                The equations are as below:
                    PE(pos, 2i) = sin(pos / 10000^{2i / d_model}), i ∈ {0, ..., d_model / 2 - 1}
                    PE(pos, 2i + 1) = cos(pos / 10000^{2i / d_model}), i ∈ {0, ..., d_model / 2 - 1}
                    Reference:
                        'Attention Is All You Need'
                        https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
                For the 'sep' type, sin is applied to the first half of dimensions and cos is applied to the second half
                of dimensions. The equations are as below:
                    PE(pos, i) = sin(pos / 10000^{2i / d_model}), i ∈ {0, ..., d_model / 2 - 1}
                    PE(pos, i) = cos(pos / 10000^{2i / d_model}), i ∈ {d_model / 2, ..., d_model - 1}
                    Reference:
                        'Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition'
                        https://ieeexplore.ieee.org/abstract/document/8462506/
            d_model: int
                The dimension of the hidden feature vectors of the Transformer layers.
            emb_scale: bool
                Controls whether the embedding vectors are scaled up by sqrt(d_model) before adding into the positional
                encoding or not.
                References:
                    Section 3.4 in 'Attention Is All You Need'
                    https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
                In most cases, we don't recommend you to turn it on especially when you don't have a large training set
                (e.g. LibriSpeech-train_clean_100) because it may make your model hard to converge. Please consider it
                only when you want to emphasize the embedded features over the positional encodings.
            emb_layernorm: bool
                Controls whether the embedding vectors are normalized by LayerNorm before adding into the positional
                encoding or not.
            posenc_scale: bool
                Controls whether the positional encodings are scaled up by a trainable scalar before adding into the
                embedded features or not.
                Reference:
                    'Neural Speech Synthesis with Transformer Network'
                    https://ojs.aaai.org/index.php/AAAI/article/view/4642/4520
            init_alpha: float
                The initial value of the alpha used for positional encoding scaling.
                Only effective when posenc_scale is True.
            max_len: int
                The maximum length of the input feature sequences.
            dropout: float
                The dropout rate for the Dropout layer after adding the positional encoding to the input
        """

        assert posenc_type in ['mix', 'sep'], \
            f"The type of PositionalEncoding layer must be either 'mix' or 'sep', but got type={posenc_type}!"
        assert d_model % 2 == 0, \
            f"Cannot apply sin/cos positional encoding to the vectors with odd dimensions (got d_model={d_model:d})."

        self.posenc_type = posenc_type
        self.d_model = d_model
        self.emb_scale = emb_scale
        if emb_layernorm:
            self.emb_layernorm = torch.nn.LayerNorm(d_model)

        self.init_alpha = init_alpha if isinstance(init_alpha, float) else float(init_alpha)
        if posenc_scale:
            self.alpha = torch.nn.Parameter(torch.tensor(self.init_alpha))

        # positional encoding matrix
        self.update_posenc(max_len)

        # positional encoding Dropout layer
        self.dropout = torch.nn.Dropout(p=dropout)

    def reset_parameters(self):
        """
        Make sure that the scalar value is not influenced by different model initialization methods.
        """
        if hasattr(self, 'alpha'):
            self.alpha.data = torch.tensor(self.init_alpha)

    def update_posenc(self, max_len: int):
        """

        Args:
            max_len:

        """

        # positional encoding calculation
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float) * (math.log(10000.0) / self.d_model)
        )
        posenc = torch.zeros(max_len, self.d_model)

        # 'mix' positional encoding: sine functions and cosine functions mix up with each other
        if self.posenc_type == 'mix':
            posenc[:, 0::2] = torch.sin(position / div_term)
            posenc[:, 1::2] = torch.cos(position / div_term)
        # 'sep' positional encoding: sine functions and cosine functions occupy the positional encoding separately
        elif self.posenc_type == 'sep':
            div_term_ext = torch.exp(
                torch.arange(self.d_model, self.d_model * 2, 2, dtype=torch.float) * (math.log(10000.0) / self.d_model)
            )
            posenc[:, :int(self.d_model / 2)] = torch.sin(position / div_term)
            posenc[:, int(self.d_model / 2):] = torch.cos(position / div_term_ext)

        # posenc = posenc.unsqueeze(0) does not put posenc into the buffer
        # here register_buffer() allows posenc to be automatically put onto GPUs as a buffer member
        self.register_buffer('posenc', posenc.unsqueeze(0))

    def forward(self, emb_feat: torch.Tensor):
        """
        Embedded feature
            -> LayerNorm(Embedded feature)
                -> LayerNorm(Embedded feature) * sqrt(d_model)
                    -> LayerNorm(Embedded feature) * sqrt(d_model) + Positional Encoding * learnable scalar
                        -> Dropout(LayerNorm(Embedded feature) * sqrt(d_model) + Positional Encoding * learnable scalar)

        Args:
            emb_feat: (batch_size, seq_len, d_model)
                Embedded input feature sequences

        Returns:
            Embedded input feature sequences with positional encoding

        """
        # in case that the input sequence is longer than the preset max_len
        if emb_feat.size(1) > self.posenc.size(1):
            self.update_posenc(emb_feat.size(1), self.d_model)

        # 1. (optional) normalize the embedded feature by LayerNorm
        if hasattr(self, 'emb_layernorm'):
            emb_feat = self.emb_layernorm(emb_feat)

        # 2. (optional) scale the embedded feature up by sqrt(d_model)
        if self.emb_scale:
            emb_feat *= math.sqrt(self.d_model)

        # 3. (optional) scale the positional encoding vectors
        posenc = self.posenc[:, :emb_feat.size(1)]
        if hasattr(self, 'alpha'):
            # avoid posenc *= self.alpha to protect the original positional encoding
            posenc = posenc * self.alpha

        # 4. (mandatory) add positional encoding into embedded feature and apply the dropout
        return self.dropout(emb_feat + posenc)

    def get_recordable_para(self) -> Dict or None:
        if hasattr(self, 'alpha'):
            return dict(alpha=self.alpha)
        else:
            return None

    def extra_repr(self) -> str:
        return f"emb_scale={self.emb_scale}\n" \
               f"posenc_scale={hasattr(self, 'alpha')}"
