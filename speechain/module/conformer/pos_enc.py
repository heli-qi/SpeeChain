import math

import torch

from speechain.module.abs import Module

class RelPositionalEncoding(Module):
    """

    """

    def module_init(self,
                    posenc_type: str = 'mix',
                    d_model: int = 512,
                    emb_scale: bool = False,
                    max_len: int = 5000,
                    dropout: float = 0.0):

        assert posenc_type in ['mix', 'sep'], \
            f"The type of PositionalEncoding layer must be either 'mix' or 'sep', but got type={posenc_type}!"
        assert d_model % 2 == 0, \
            f"Cannot apply sin/cos positional encoding to the vectors with odd dimensions (got d_model={d_model:d})."

        self.posenc_type = posenc_type
        self.d_model = d_model
        self.emb_scale = emb_scale

        # positional encoding matrix
        self.max_len = max_len
        self.update_posenc(max_len)

        # positional encoding Dropout layer
        self.dropout = torch.nn.Dropout(p=dropout)

    def update_posenc(self, seq_len: int = None):
        """

        Args:
            max_len:

        """
        if seq_len is None:
            seq_len = self.max_len

        if seq_len >= self.max_len:
            # positional encoding calculation
            position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float) * (math.log(10000.0) / self.d_model)
            )
            posenc_past, posenc_future = torch.zeros(seq_len, self.d_model), torch.zeros(seq_len, self.d_model)

            # 'mix' positional encoding: sine functions and cosine functions mix up with each other
            if self.posenc_type == 'mix':
                posenc_past[:, 0::2] = torch.sin(position / div_term)
                posenc_past[:, 1::2] = torch.cos(position / div_term)

                posenc_future[:, 0::2] = torch.sin(-1 * position / div_term)
                posenc_future[:, 1::2] = torch.cos(-1 * position / div_term)
            # 'sep' positional encoding: sine functions and cosine functions occupy the positional encoding separately
            elif self.posenc_type == 'sep':
                div_term_ext = torch.exp(
                    torch.arange(self.d_model, self.d_model * 2, 2, dtype=torch.float) * (math.log(10000.0) / self.d_model)
                )
                posenc_past[:, :int(self.d_model / 2)] = torch.sin(position / div_term)
                posenc_past[:, int(self.d_model / 2):] = torch.cos(position / div_term_ext)

                posenc_future[:, :int(self.d_model / 2)] = torch.sin(-1 * position / div_term)
                posenc_future[:, int(self.d_model / 2):] = torch.cos(-1 * position / div_term_ext)

            # posenc = posenc.unsqueeze(0) does not put posenc into the buffer
            # here register_buffer() allows posenc to be automatically put onto GPUs as a buffer member
            self.register_buffer('posenc_past', posenc_past.unsqueeze(0))
            self.register_buffer('posenc_future', posenc_future.unsqueeze(0))

    def forward(self, emb_feat: torch.Tensor):

        # update the buffered positional encodings by sequence length
        seq_len = emb_feat.size(1)
        self.update_posenc(seq_len)

        with torch.no_grad():
            posenc_past = torch.flip(self.posenc_past[:, :seq_len], (1,))
            posenc_future = self.posenc_future[:, 1: seq_len]
            # (1, 2 * seq_len - 1, embed_dim)
            posenc = torch.cat([posenc_past, posenc_future], dim=1)

        # (optional) scale the embedded feature up by sqrt(d_model)
        if self.emb_scale:
            emb_feat *= math.sqrt(self.d_model)

        # apply dropout to both prenet embedded feature and positional encoding
        return self.dropout(emb_feat), self.dropout(posenc)
