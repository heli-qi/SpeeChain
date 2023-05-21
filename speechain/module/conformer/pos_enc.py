import math
import torch

from speechain.module.abs import Module
from speechain.module.transformer.pos_enc import PositionalEncoding

class RelPosEncXL(Module):
    """

    """

    def module_init(self,
                    posenc_type: str = 'mix',
                    d_model: int = 512,
                    max_len: int = 5000):

        self.abs_posenc = PositionalEncoding(
            posenc_type=posenc_type, d_model=d_model, max_len=max_len,
        )

    def forward(self, emb_feat: torch.Tensor):
        seq_len = emb_feat.size(1)
        if seq_len > self.abs_posenc.posenc.size(1):
            self.abs_posenc.update_posenc(seq_len)

        with torch.no_grad():
            pe_past = torch.flip(self.abs_posenc.posenc[:, :seq_len], (1,))
            pe_future = self.abs_posenc.posenc[:, 1: seq_len]
            pe = torch.cat([pe_past, pe_future], dim=1)
            # pe is now 1, 2*seq_len, embed_dim
            return pe