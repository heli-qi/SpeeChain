"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch

from typing import Dict
from speechain.module.abs import Module
from speechain.utilbox.train_util import make_mask_from_len

from speechain.module.prenet.embed import EmbedPrenet
from speechain.module.transformer.decoder import TransformerDecoder
from speechain.module.postnet.token import TokenPostnet


class ASRDecoder(Module):
    """

    """
    prenet_class_dict = dict(
        embed=EmbedPrenet
    )

    decoder_class_dict = dict(
        transformer=TransformerDecoder
    )

    def module_init(self, prenet: Dict, decoder: Dict, vocab_size: int = None):
        """

        Args:
            prenet:
            decoder:
            vocab_size:

        """
        # temporary register for connecting two sequential modules
        _prev_output_size = None

        # embedding layer of the E2E ASR decoder
        prenet_class = self.prenet_class_dict[prenet['type']]
        prenet['conf'] = dict() if 'conf' not in prenet.keys() else prenet['conf']
        self.prenet = prenet_class(vocab_size=vocab_size, **prenet['conf'])
        _prev_output_size = self.prenet.output_size

        # main body of the E2E ASR decoder
        decoder_class = self.decoder_class_dict[decoder['type']]
        decoder['conf'] = dict() if 'conf' not in decoder.keys() else decoder['conf']
        self.decoder = decoder_class(input_size=_prev_output_size, **decoder['conf'])
        _prev_output_size = self.decoder.output_size

        # token prediction layer for the E2E ASR decoder
        self.postnet = TokenPostnet(input_size=_prev_output_size, vocab_size=vocab_size)

    def forward(self,
                enc_feat: torch.Tensor,
                enc_feat_mask: torch.Tensor,
                text: torch.Tensor,
                text_len: torch.Tensor):
        """

        Args:
            enc_feat:
            enc_feat_mask:
            text:
            text_len:

        Returns:

        """
        # Text Embedding
        emb_text = self.prenet(text)

        # mask generation for the input text
        text_mask = make_mask_from_len(text_len)
        if text.is_cuda:
            text_mask = text_mask.cuda(text.device)

        dec_feat, self_attmat, encdec_attmat, hidden = self.decoder(
            src=enc_feat, src_mask=enc_feat_mask, tgt=emb_text, tgt_mask=text_mask)
        return self.postnet(dec_feat), self_attmat, encdec_attmat, hidden
