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
                    use_lookup: bool = False,
                    use_pretrain: bool = False,
                    spk_emb_comb: str = 'concat',
                    spk_emb_act: str = 'Softsign',
                    dec_comb: bool = False,
                    encdec_same_proj: bool = True):
        """

        Args:
            d_model:
            spk_emb_dim:
            spk_num:
            use_lookup:
            use_pretrain:
            spk_emb_comb:
            spk_emb_act:
            dec_comb:
            encdec_same_proj:

        Returns:

        """
        assert use_lookup or use_pretrain, \
            "use_lookup and use_pretrain cannot be False at the same time! Please set at least one of them to True."
        assert spk_emb_comb in ['add', 'concat'], \
            f"spk_emb_comb must be either 'add' or 'concat', but got {spk_emb_comb}."
        assert 'ReLU' not in spk_emb_act, \
            "Please don't use the activation function from 'ReLU' family for speaker embedding because they will " \
            "change the centroid of a vector."

        self.spk_emb_dim = spk_emb_dim
        # for close-set multi-speaker TTS, speaker lookup table is created for extracting embeddings from speaker IDs
        self.use_lookup = use_lookup
        if use_lookup:
            self.spk_lookup = EmbedPrenet(embedding_dim=spk_emb_dim, vocab_size=spk_num)
            # for dimension consistency with the model
            if spk_emb_comb == 'add' and spk_emb_dim != d_model:
                self.pre_add_proj_lookup = torch.nn.Linear(spk_emb_dim, d_model)

        # initialize the linear projection layer
        self.spk_emb_comb = spk_emb_comb
        # for open-set multi-speaker TTS, speaker embedding is extracted by a pretrained speaker embedding model
        self.use_pretrain = use_pretrain
        if use_pretrain:
            # for dimension consistency with the model
            if spk_emb_comb == 'add' and spk_emb_dim != d_model:
                self.pre_add_proj_pretrain = torch.nn.Linear(spk_emb_dim, d_model)

        # initialize the activation function for the speaker embedding
        self.spk_emb_act = spk_emb_act
        self.activation = getattr(torch.nn, spk_emb_act)()

        # at the end of SpeakerEmbedPrenet, there is a linear projection layer shared by both open-set and close-set
        # multi-speaker TTS models before passing the results to the TTS decoder
        proj_in_size = d_model
        if use_lookup and spk_emb_comb == 'concat':
            proj_in_size += spk_emb_dim
        if use_pretrain and spk_emb_comb == 'concat':
            proj_in_size += spk_emb_dim
        self.final_proj_enc = torch.nn.Linear(proj_in_size, d_model)

        # the projection layer can also be applied to the input of AR-TTS decoder
        self.dec_comb = dec_comb
        self.encdec_same_proj = encdec_same_proj
        if self.dec_comb and not self.encdec_same_proj:
            self.final_proj_dec = torch.nn.Linear(proj_in_size, d_model)

    def forward(self, spk_ids: torch.Tensor = None, spk_feat: torch.Tensor = None, spk_feat_act: bool = True):
        """

        Args:
            spk_ids:
            spk_feat:
            spk_feat_act:

        Returns:

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

        # 3. normalize the speaker feature to make its centroid zero
        if spk_feat_act:
            spk_feat_lookup = self.activation(spk_feat_lookup) if spk_feat_lookup is not None else spk_feat_lookup
            spk_feat = self.activation(spk_feat) if spk_feat is not None else spk_feat

        return spk_feat_lookup, spk_feat

    def combine_spk_feat(self, spk_feat: torch.Tensor, spk_feat_lookup: torch.Tensor,
                         enc_output: torch.Tensor, dec_input: torch.Tensor = None):
        """

        Args:
            spk_feat: (batch, spk_feat_dim) or (batch, 1, spk_feat_dim)
            spk_feat_lookup: (batch, spk_feat_dim) or (batch, 1, spk_feat_dim)
            enc_output: (batch, enc_maxlen, d_model)
            dec_input: (batch, dec_maxlen, d_model)

        Returns: (batch, seq_len, d_model)

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
