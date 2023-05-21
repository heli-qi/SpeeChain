from typing import Dict

import torch

from speechain.module.abs import Module
from speechain.module.prenet.embed import EmbedPrenet


class SpeakerEmbedPrenet(Module):
    """
        SpeakerEmbedPrenet is a module for integrating speaker embeddings into a TTS model.
        It supports both open-set and close-set multi-speaker TTS and can combine speaker embeddings with encoder
        and/or decoder inputs.
    """

    def module_init(self,
                    d_model: int,
                    spk_emb_dim_lookup: int = None,
                    spk_emb_dim_pretrained: int = None,
                    spk_num: int = None,
                    spk_emb_comb: str = 'concat',
                    dec_comb: bool = False,
                    encdec_same_proj: bool = True):
        """
            Initialize the SpeakerEmbedPrenet module with the given configuration.

            Args:
                d_model: int
                    input/output feature dimension for the TTS model
                spk_emb_dim_lookup: int
                    speaker embedding dimension for close-set TTS with lookup table (optional)
                spk_emb_dim_pretrained: int
                    speaker embedding dimension for open-set TTS with pretrained embeddings (optional)
                spk_num: int
                    number of speakers for close-set TTS (optional)
                spk_emb_comb: str
                    method for combining speaker embeddings with encoder/decoder inputs, either 'add' or 'concat'
                dec_comb: bool
                    whether to combine speaker embeddings with decoder inputs
                encdec_same_proj: bool
                    whether to use the same projection layer for encoder and decoder

            Returns:
                None
        """
        self.use_lookup, self.use_pretrain = spk_emb_dim_lookup is not None, spk_emb_dim_pretrained is not None
        assert self.use_lookup or self.use_pretrain, \
            "spk_emb_dim_lookup and spk_emb_dim_pretrained cannot be None at the same time! " \
            "Please specify the value of at least one of them."
        assert spk_emb_comb in ['add', 'concat'], \
            f"spk_emb_comb must be either 'add' or 'concat', but got {spk_emb_comb}."

        # for close-set multi-speaker TTS, speaker lookup table is created for extracting embeddings from speaker IDs
        if self.use_lookup:
            assert spk_emb_dim_lookup is not None
            self.spk_emb_dim_lookup = spk_emb_dim_lookup
            self.spk_lookup = EmbedPrenet(embedding_dim=self.spk_emb_dim_lookup, vocab_size=spk_num)
            # for dimension consistency with the model
            if spk_emb_comb == 'add' and self.spk_emb_dim_lookup != d_model:
                self.pre_add_proj_lookup = torch.nn.Linear(self.spk_emb_dim_lookup, d_model)

        # initialize the linear projection layer
        self.spk_emb_comb = spk_emb_comb
        self.spk_emb_dim_pretrained = spk_emb_dim_pretrained
        # for open-set multi-speaker TTS, speaker embedding is extracted by a pretrained speaker embedding model
        if self.use_pretrain:
            assert spk_emb_dim_pretrained is not None
            self.spk_emb_dim_pretrained = spk_emb_dim_pretrained
            # for dimension consistency with the model
            if spk_emb_comb == 'add' and self.spk_emb_dim_pretrained != d_model:
                self.pre_add_proj_pretrain = torch.nn.Linear(self.spk_emb_dim_pretrained, d_model)

        # at the end of SpeakerEmbedPrenet, there is a linear projection layer shared by both open-set and close-set
        # multi-speaker TTS models before passing the results to the TTS decoder
        proj_in_size = d_model
        if self.use_lookup and spk_emb_comb == 'concat':
            proj_in_size += self.spk_emb_dim_lookup
        if self.use_pretrain and spk_emb_comb == 'concat':
            proj_in_size += self.spk_emb_dim_pretrained
        self.final_proj_enc = torch.nn.Linear(proj_in_size, d_model)

        # the projection layer can also be applied to the input of AR-TTS decoder
        self.dec_comb = dec_comb
        self.encdec_same_proj = encdec_same_proj
        if self.dec_comb and not self.encdec_same_proj:
            self.final_proj_dec = torch.nn.Linear(proj_in_size, d_model)

    def forward(self, spk_ids: torch.Tensor = None, spk_feat: torch.Tensor = None):
        """
            Forward pass of the SpeakerEmbedPrenet module to obtain speaker features.

            Args:
                spk_ids: torch.Tensor
                    speaker IDs for close-set TTS (optional)
                spk_feat: torch.Tensor
                    pretrained speaker embeddings for open-set TTS (optional)

            Returns:
                tuple: speaker features from lookup table and pretrained embeddings
        """
        # 1. extract the speaker feature from the embedding look-up table
        if self.use_lookup:
            assert spk_ids is not None, "For lookup-based close-set TTS, you must pass spk_ids to SpeakerEmbedPrenet."
            spk_feat_lookup = self.spk_lookup(spk_ids)
            # 1.2. project the lookup speaker feature to the same dimension with the model input
            if hasattr(self, 'pre_add_proj_lookup'):
                spk_feat_lookup = self.pre_add_proj_lookup(spk_feat_lookup)
        else:
            spk_feat_lookup = None

        # 2. project pretrained speaker feature to the same dimension with the model input
        if self.use_pretrain:
            assert spk_feat is not None, "For pretrained-bsed open-set TTS, you must pass spk_feat to SpeakerEmbedPrenet."
            if hasattr(self, 'pre_add_proj_pretrain'):
                spk_feat = self.pre_add_proj_pretrain(spk_feat)
        else:
            spk_feat = None

        # 3. activate speaker embeddings before fusion
        # process the lookup speaker embeddings by non-linear function softsign
        spk_feat_lookup = torch.nn.functional.normalize(spk_feat_lookup, dim=-1) if spk_feat_lookup is not None else spk_feat_lookup
        # normalize the pretrained speaker embeddings to transform it to the unit sphere
        spk_feat = torch.nn.functional.normalize(spk_feat, dim=-1) if spk_feat is not None else spk_feat

        return spk_feat_lookup, spk_feat

    def combine_spk_feat(self, spk_feat: torch.Tensor, spk_feat_lookup: torch.Tensor,
                         enc_output: torch.Tensor, dec_input: torch.Tensor = None):
        """
            Combine speaker features with TTS model's encoder and/or decoder inputs.

            Args:
                spk_feat: torch.Tensor
                    pretrained speaker embeddings for open-set TTS (optional)
                spk_feat_lookup: torch.Tensor
                    speaker features from lookup table for close-set TTS (optional)
                enc_output: torch.Tensor
                    TTS encoder output
                dec_input: torch.Tensor
                    TTS decoder input (optional)

            Returns:
                tuple: TTS encoder output and decoder input combined with speaker features
        """
        if spk_feat is not None:
            if spk_feat.dim() == 2:
                spk_feat = spk_feat.unsqueeze(1)
            else:
                assert spk_feat.dim() == 3 and spk_feat.size(1) == 1, \
                    f"Something wrong happens to the dimension of the input spk_feat. Its dimension is {spk_feat.shape}."

        if spk_feat_lookup is not None:
            if spk_feat_lookup.dim() == 2:
                spk_feat_lookup = spk_feat_lookup.unsqueeze(1)
            else:
                assert spk_feat_lookup.dim() == 3 and spk_feat_lookup.size(1) == 1, \
                    f"Something wrong happens to the dimension of the input spk_feat_lookup. " \
                    f"Its dimension is {spk_feat_lookup.shape}."

        def combine_spk_feat_to_tgt(tgt_feat: torch.Tensor, proj_layer: torch.nn.Linear):
            # directly add the speaker embedding into the target features
            if self.spk_emb_comb == 'add':
                if spk_feat is not None:
                    tgt_feat = tgt_feat + spk_feat
                if spk_feat_lookup is not None:
                    tgt_feat = tgt_feat + spk_feat_lookup
            # concatenate the target features with the speaker features in the last dimension
            elif self.spk_emb_comb == 'concat':
                if spk_feat is not None:
                    tgt_feat = torch.cat([tgt_feat, spk_feat.expand(-1, tgt_feat.size(1), -1)], dim=-1)
                if spk_feat_lookup is not None:
                    tgt_feat = torch.cat([tgt_feat, spk_feat_lookup.expand(-1, tgt_feat.size(1), -1)], dim=-1)
            else:
                raise RuntimeError
            # project the concatenated vectors to the same dimension as self.d_model
            return proj_layer(tgt_feat)

        # (mandatory) combine the speaker embedding to the TTS encoder outputs
        enc_output = combine_spk_feat_to_tgt(enc_output, self.final_proj_enc)
        # (optional) combine the speaker embedding to the TTS decoder inputs
        if self.dec_comb:
            assert dec_input is not None, \
                "If you want to combine speaker embeddings with decoder input vectors, " \
                "please give the decoder input vectors by the argument 'dec_input' in combine_spk_feat()"
            dec_input = combine_spk_feat_to_tgt(dec_input,
                                                self.final_proj_enc if self.encdec_same_proj else self.final_proj_dec)

        return enc_output, dec_input

    def extra_repr(self) -> str:
        return f"dec_comb={self.dec_comb}"
