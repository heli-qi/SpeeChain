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

        # Decoding
        dec_results = self.decoder(src=enc_feat, src_mask=enc_feat_mask,
                                   tgt=emb_text, tgt_mask=text_mask)

        # initialize the decoder outputs
        dec_outputs = dict(
            output=self.postnet(dec_results['output'])
        )
        # if the build-in decoder has the attention results
        if 'att' in dec_results.keys():
            dec_outputs.update(
                att=dec_results['att']
            )
        # if the build-in decoder has the hidden results
        if 'hidden' in dec_results.keys():
            dec_outputs.update(
                hidden=dec_results['hidden']
            )
        return dec_outputs


    def get_trainable_scalars(self) -> Dict or None:
        trainable_scalars = dict()
        # decoder-prenet layers
        pre_scalars = self.prenet.get_trainable_scalars()
        if pre_scalars is not None:
            trainable_scalars.update(**pre_scalars)
        # decoder layers
        dec_scalars = self.decoder.get_trainable_scalars()
        if dec_scalars is not None:
            trainable_scalars.update(**dec_scalars)
        # decoder-postnet layers
        post_scalars = self.postnet.get_trainable_scalars()
        if post_scalars is not None:
            trainable_scalars.update(**post_scalars)

        return trainable_scalars
