import torch
from torch import nn

from speechain.module.abs import Module
from speechain.module.conformer.pos_enc import RelPosEncXL
from speechain.module.conformer.attention import RelPosMHAXL


class ConvolutionModule(Module):

    def module_init(self, **module_conf):
        pass

    def forward(self, **kwargs):
        pass


class ConformerEncoderLayer(Module):

    def module_init(self, **module_conf):
        pass

    def forward(self, **kwargs):
        pass


class ConformerEncoder(Module):
    """

    """

    def module_init(self,
                    posenc_type: str = 'mix',
                    d_model: int = 512,
                    num_heads: int = 4,
                    num_layers: int = 8):
        # input_size and output_size initialization
        if self.input_size is not None:
            d_model = self.input_size
        self.output_size = d_model

        # para recording
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.posenc = RelPosEncXL(posenc_type=posenc_type, d_model=d_model)

        # initialize transformer layers
        self.cfm_layers = torch.nn.ModuleList([
            ConformerEncoderLayer(d_model=d_model,
                                  num_heads=num_heads,
                                  att_dropout=att_dropout,
                                  fdfwd_dim=fdfwd_dim,
                                  fdfwd_activation=fdfwd_activation,
                                  fdfwd_dropout=fdfwd_dropout,
                                  res_dropout=res_dropout)
            for _ in range(num_layers)])

        # initialize layernorm layer
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src: torch.Tensor, mask: torch.Tensor):
        # add position encoding to word embeddings
        rel_posenc = self.posenc(src)

        return src, mask