from typing import Dict

import torch

from speechain.module.abs import Module
from speechain.module.prenet.embed import EmbedPrenet


class SpeakerEmbedPrenet(Module):
    """

    """

    def module_init(self,
                    d_model: int,
                    spk_emb_dim: int,
                    spk_num: int = None,
                    spk_emb_mode: str = 'close',
                    spk_emb_comb: str = 'concat',
                    spk_emb_act: str = 'Softsign',
                    spk_emb_scale: bool = False,
                    dec_comb: bool = False,
                    same_proj: bool = True):
        """

        Args:
            d_model:
            spk_emb_dim:
            spk_num:
            spk_emb_mode:
            spk_emb_comb:
            spk_emb_act:
            spk_emb_scale:
            dec_comb:
            same_proj:

        Returns:

        """
        assert spk_emb_mode in ['open', 'close'], \
            f"spk_emb_mode must be either 'open' or 'close', but got {spk_emb_mode}."
        assert spk_emb_comb in ['add', 'concat'], \
            f"spk_emb_comb must be either 'add' or 'concat', but got {spk_emb_comb}."
        assert 'ReLU' not in spk_emb_act, \
            "Please don't use the activation function from 'ReLU' family for speaker embedding because they will " \
            "change the centroid of a vector."

        self.spk_emb_dim = spk_emb_dim
        # for close-set multi-speaker TTS, speaker embedding is jointly trained with TTS model
        if spk_emb_mode == 'close':
            self.spk_embed = EmbedPrenet(embedding_dim=spk_emb_dim, vocab_size=spk_num)

        # initialize the linear projection layer
        self.spk_emb_comb = spk_emb_comb
        if spk_emb_comb == 'add' and spk_emb_dim != d_model:
            self.pre_add_proj = torch.nn.Linear(spk_emb_dim, d_model)

        # initialize the activation function for the speaker embedding
        self.activation = getattr(torch.nn, spk_emb_act)()

        # initialize the speaker embedding scalar
        if spk_emb_scale:
            self.alpha = torch.nn.Parameter(torch.tensor(1.0))

        # at the end of SpeakerEmbedPrenet, there is a linear projection layer shared by both open-set and close-set
        # multi-speaker TTS models before passing the results to the TTS decoder
        self.dec_comb = dec_comb
        self.same_proj = same_proj
        self.final_proj = torch.nn.Linear(d_model if spk_emb_comb == 'add' else spk_emb_dim + d_model, d_model)
        if self.dec_comb and not self.same_proj:
            self.final_proj_dec = torch.nn.Linear(d_model if spk_emb_comb == 'add' else spk_emb_dim + d_model, d_model)

    def reset_parameters(self):
        """
        Make sure that the scalar value is not influenced by different model initialization methods.
        """
        if hasattr(self, 'alpha'):
            self.alpha.data = torch.tensor(1.0)

    def forward(self, spk_ids: torch.Tensor = None, spk_feat: torch.Tensor = None, spk_feat_act: bool = True):
        """

        Args:
            spk_ids:
            spk_feat:
            no_spk_feat_act:

        Returns:

        """
        assert (spk_ids is not None) ^ (spk_feat is not None), \
            "You could only pass one of spk_ids and spk_feat to SpeakerEmbedPrenet. " \
            "If you want to train an open-set multi-speaker TTS, please given spk_feat; " \
            "If you want to train a close-set multi-speaker TTS, please given spk_ids."

        # 1. (mandatory for close-set multi-speaker TTS) extract the speaker feature from the embedding look-up table
        if hasattr(self, 'spk_embed') and spk_ids is not None:
            spk_feat = self.spk_embed(spk_ids)

        # 2. (mandatory for adding combination) project speaker feature to the same dimension with the model input
        if hasattr(self, 'pre_add_proj'):
            spk_feat = self.pre_add_proj(spk_feat)

        # 3. (mandatory) normalize the speaker feature to make its centroid zero
        if spk_feat_act:
            spk_feat = self.activation(spk_feat)

        # 4. (optional) scale the speaker embedding vectors
        if hasattr(self, 'alpha'):
            spk_feat = spk_feat * self.alpha

        return spk_feat

    def combine_spk_feat(self, spk_feat: torch.Tensor, enc_output: torch.Tensor, dec_input: torch.Tensor):
        """

        Args:
            spk_feat: (batch, spk_feat_dim) or (batch, 1, spk_feat_dim)
            enc_output: (batch, enc_maxlen, d_model)
            dec_input: (batch, dec_maxlen, d_model)

        Returns: (batch, seq_len, d_model)

        """
        # check the shape of the speaker features, the second dim must be 1 for broadcasting to the target features
        if spk_feat.dim() == 2:
            spk_feat = spk_feat.unsqueeze(1)
        else:
            assert spk_feat.dim() == 3 and spk_feat.size(1) == 1, \
                f"Something wrong happens to the dimension of the input spk_feat. Its dimension is {spk_feat.shape}."

        def combine_spk_feat_to_tgt(tgt_feat: torch.Tensor, proj_layer: torch.nn.Linear):
            # directly add the speaker embedding into the target features
            if self.spk_emb_comb == 'add':
                tgt_feat = spk_feat + tgt_feat
            # concatenate the target features with the speaker features in the last dimension
            elif self.spk_emb_comb == 'concat':
                tgt_feat = torch.cat([tgt_feat, spk_feat.expand(-1, tgt_feat.size(1), -1)], dim=-1)
            else:
                raise RuntimeError
            # project the concatenated vectors to the same dimension as self.d_model
            return proj_layer(tgt_feat)

        # (mandatory) combine the speaker embedding to the TTS encoder outputs
        enc_output = combine_spk_feat_to_tgt(enc_output, self.final_proj)
        # (optional) combine the speaker embedding to the TTS decoder inputs
        if self.dec_comb:
            dec_input = combine_spk_feat_to_tgt(dec_input, self.final_proj if self.same_proj else self.final_proj_dec)

        return enc_output, dec_input

    def get_trainable_scalars(self) -> Dict or None:
        if hasattr(self, 'alpha'):
            return dict(
                alpha=self.alpha.data
            )
        else:
            return None

    def extra_repr(self) -> str:
        output = f"spk_emb_scale={hasattr(self, 'alpha')}\n" \
                 f"dec_comb={self.dec_comb}\n" \
                 f"same_proj={self.same_proj}"
        return output
