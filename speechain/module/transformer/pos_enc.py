"""
    Origin: Sashi Novitasari
    Modification: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import math
import torch
import torch.nn as nn

from speechain.module.abs import Module

class PositionalEncoding(Module):
    """
    Pre-compute position encodings (PE). In forward pass, this module adds the positional encodings to the embedded
    feature vectors to make the Transformer aware of the positional information of the sequences.

    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def module_init(self,
                    type: str = 'mix',
                    d_model: int = 512,
                    max_len: int = 5000,
                    dropout: float = 0.0):
        """
        Positional Encoding with maximum length max_len.

        Args:
            type: str
                The type of positional encoding (must be either 'cross' or 'order').
                For the 'mix' type, sin is applied to the odd dimensions and cos is applied to the even dimensions.
                The equations are as below:
                    PE(pos, 2i) = sin(pos / 10000^{2i / d_model}), i ∈ {0, ..., d_model / 2 - 1}
                    PE(pos, 2i + 1) = cos(pos / 10000^{2i / d_model}), i ∈ {0, ..., d_model / 2 - 1}
                    Reference:
                        'Attention Is All You Need' (https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
                For the 'sep' type, sin is applied to the first half of dimensions and cos is applied to the second half of dimensions.
                The equations are as below:
                    PE(pos, i) = sin(pos / 10000^{2i / d_model}), i ∈ {0, ..., d_model / 2 - 1}
                    PE(pos, i) = cos(pos / 10000^{2i / d_model}), i ∈ {d_model / 2, ..., d_model - 1}
                    Reference:
                        'Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition' (https://ieeexplore.ieee.org/abstract/document/8462506/)
            d_model: int
                The dimension of the hidden feature vectors of the Transformer layers.
            max_len: int
                The maximum length of the input feature sequences.
            dropout: float
                The dropout rate for the Dropout layer after adding the positional encoding to the input
        """

        assert type in ['mix', 'sep'], \
            f"The type of PositionalEncoding layer must be either 'cross' or 'order', but got type={type}!"
        assert d_model % 2 == 0, \
            f"Cannot apply sin/cos positional encoding to the vectors with odd dimensions (got d_model={d_model:d})."

        # positional encoding calculation
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
        posenc = torch.zeros(max_len, d_model)

        if type == 'mix':
            posenc[:, 0::2] = torch.sin(position * div_term)
            posenc[:, 1::2] = torch.cos(position * div_term)

        elif type == 'sep':
            div_term_ext = torch.exp((torch.arange(d_model, d_model * 2, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
            posenc[:, : int(d_model / 2)] = torch.sin(position * div_term)
            posenc[:, int(d_model / 2):] = torch.cos(position * div_term_ext)

        self.register_buffer('posenc', posenc.unsqueeze(0))
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, emb_feat):
        """
        Embed inputs.

        Args:
            emb_feat: (batch_size, seq_len, d_model)
                Embedded input feature sequences

        Returns:
            Embedded input feature sequences with positional encoding

        """

        assert emb_feat.size(1) <= self.posenc.size(1), \
            f"The length of the input features is longer than max_len!" \
            f"max_len={self.posenc.size(1):d}, but got {emb_feat.size(1):d}"

        return self.dropout(emb_feat + self.posenc[:, :emb_feat.size(1)])
