"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch

from typing import Dict
from speechain.module.abs import Module
from speechain.utilbox.import_util import import_class
from speechain.utilbox.train_util import make_mask_from_len

class ASRDecoder(Module):
    """

    """
    def module_init(self, prenet: Dict, decoder: Dict, postnet: Dict, vocab_size: int = None):
        """

        Args:
            prenet:
            decoder:
            postnet:
            vocab_size:

        Returns:

        """
        _prev_output_size = None

        prenet_class = import_class('speechain.module.' + prenet['type'])
        prenet['conf'] = dict() if 'conf' not in prenet.keys() else prenet['conf']
        self.prenet = prenet_class(vocab_size=vocab_size, **prenet['conf'])
        _prev_output_size = self.prenet.output_size

        decoder_class = import_class('speechain.module.' + decoder['type'])
        decoder['conf'] = dict() if 'conf' not in decoder.keys() else decoder['conf']
        self.decoder = decoder_class(input_size=_prev_output_size, **decoder['conf'])
        _prev_output_size = self.decoder.output_size

        postnet_class = import_class('speechain.module.' + postnet['type'])
        postnet['conf'] = dict() if 'conf' not in postnet.keys() else postnet['conf']
        self.postnet = postnet_class(input_size=_prev_output_size, vocab_size=vocab_size, **postnet['conf'])


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