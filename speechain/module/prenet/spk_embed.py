from typing import Dict

import torch

from speechain.module.abs import Module
from speechain.module.prenet.embed import EmbedPrenet

class SpeakerEmbedPrenet(Module):
    """

    """
    def module_init(self,
                    spk_emb_dim: int,
                    d_model: int,
                    spk_num: int = None,
                    spk_emb_mode: str = 'close',
                    spk_emb_comb: str = 'concat',
                    spk_emb_act: str = 'Softsign',
                    spk_emb_scale: bool = False):
        """

        Args:
            spk_emb_dim:
            d_model:
            spk_num:
            spk_emb_mode:
            spk_emb_comb:
            spk_emb_act:
            spk_emb_scale:

        """
        # initialize the embedding layer
        self.spk_emb_dim = spk_emb_dim
        if spk_emb_mode == 'close':
            self.embedding = EmbedPrenet(embedding_dim=spk_emb_dim, vocab_size=spk_num)
        elif spk_emb_mode != 'open':
            raise ValueError

        # initialize the linear projection layer
        self.spk_emb_comb = spk_emb_comb
        self.lnr_proj = torch.nn.Linear(spk_emb_dim if spk_emb_comb == 'add' else spk_emb_dim + d_model, d_model)

        # initialize the final activation function
        if spk_emb_act == 'normalize':
            self.activation = torch.nn.functional.normalize
        else:
            self.activation = getattr(torch.nn, spk_emb_act)()

        # initialize the speaker embedding scalar
        if spk_emb_scale:
            self.spk_emb_scalar = torch.nn.Parameter(torch.tensor(1.0))


    def reset_parameters(self):
        """
        Make sure that the scalar value is not influenced by different model initialization methods.
        """
        if hasattr(self, 'spk_emb_scalar'):
            self.spk_emb_scalar.data = torch.tensor(1.0)


    def forward(self, spk_ids: torch.Tensor = None, spk_feat: torch.Tensor = None):
        """

        Args:
            spk_ids: (batch,)
            spk_feat: (batch, spk_feat_dim)
            batch_size:
            device:

        Returns:

        """
        assert (spk_ids is not None) ^ (spk_feat is not None)
        # 1. (mandatory for close-set multi-speaker TTS) extract the speaker feature from the embedding look-up table
        if hasattr(self, 'embedding') and spk_ids is not None:
            spk_feat = self.embedding(spk_ids)

        # 2. (mandatory for adding combination) project speaker feature to the same dimension with the model input
        if self.spk_emb_comb == 'add':
            spk_feat = self.lnr_proj(spk_feat)

        # 3. convert the speaker feature to the range (-1, 1) to fit the normalized encoder output and decoder input
        spk_feat = self.activation(spk_feat)

        # 4. (optional) scale the speaker embedding vectors
        if hasattr(self, 'spk_emb_scalar'):
            spk_feat = spk_feat * self.spk_emb_scalar

        return spk_feat


    def combine_spk_emb(self, spk_feat: torch.Tensor, tgt: torch.Tensor):
        """

        Args:
            spk_feat: (batch, spk_feat_dim) or (batch, 1, spk_feat_dim)
            tgt: (batch, seq_len, d_model)

        Returns: (batch, seq_len, d_model)

        """
        # check the shape of the speaker features, the second dim must be 1 for broadcasting to the target features
        if spk_feat.dim() == 2:
            spk_feat = spk_feat.unsqueeze(1)
        else:
            assert spk_feat.dim() == 3 and spk_feat.size(1) == 1

        # directly return the summation of the target features and the speaker features
        if self.spk_emb_comb == 'add':
            return spk_feat + tgt

        # concatenate the target features with the speaker features in the last dimension
        tgt_feat = torch.cat([tgt, spk_feat.expand(-1, tgt.size(1), -1)], dim=-1)
        # project the concatenated vectors to the same dimension as the model input
        return self.lnr_proj(tgt_feat)


    def get_trainable_scalars(self) -> Dict or None:
        if hasattr(self, 'spk_emb_scalar'):
            return dict(
                spk_emb_scalar=self.spk_emb_scalar.data
            )
        else:
            return None


    def extra_repr(self) -> str:
        output = f"spk_emb_scale={hasattr(self, 'spk_emb_scalar')}"
        if not isinstance(self.activation, torch.nn.Module):
            output += "\nactivation=torch.nn.functional.normalize"
        return output
