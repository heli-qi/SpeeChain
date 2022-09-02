"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch
from torch.cuda.amp import autocast
from typing import Dict
from speechain.module.abs import Module
from speechain.utilbox.import_util import import_class
from speechain.utilbox.train_util import make_mask_from_len

class TTSEncoder(Module):
    """
    TTs encoder for autoregressive decoder
    """
    def module_init(self,
                    encoder: Dict,
                    embedding: Dict,
                    prenet: Dict = None,
                    vocab_size: int = None):
        """

        Args:
            encoder:
            embedding:
            prenet:
            vocab_size:
        Returns:

        """
        #Token embedding
        embedding_class = import_class('speechain.module.' + embedding['type'])
        embedding['conf'] = dict() if 'conf' not in embedding.keys() else embedding['conf']
        self.emb = embedding_class(**embedding['conf'])
        _prev_output_size = embedding['conf']['embedding_dim']

        #Prenet
        if prenet is not None:
            prenet_class = import_class('speechain.module.' + prenet['type'])
            prenet['conf'] = dict() if 'conf' not in prenet.keys() else prenet['conf']
            self.prenet = prenet_class(input_size=_prev_output_size, **prenet['conf'])
            _prev_output_size = self.prenet.output_size

        #Encoder
        encoder_class = import_class('speechain.module.' + encoder['type'])
        encoder['conf'] = dict() if 'conf' not in encoder.keys() else encoder['conf']
        self.encoder = encoder_class(input_size=_prev_output_size, **encoder['conf'])

    def forward(self, text: torch.Tensor, text_len: torch.Tensor):
        """

        Args:
            text: (torch size: batch, maxlen)
            text_len: (torch size: batch )

        Returns:
            enc_outputs: dict
        """


        #Embedding
        text = self.emb(text)

        #Prenet
        if hasattr(self, 'prenet'):
            text, text_len = self.prenet(text, text_len)

        # mask generation for the embedded textures
        text_mask = make_mask_from_len(text_len)
        if text.is_cuda:
            text_mask = text_mask.cuda(text.device)

        #Encoder
        enc_results = self.encoder(text, text_mask)

        # initialize the outputs
        enc_outputs = dict(
            enc_feat=enc_results['output'],
            enc_feat_mask=enc_results['mask']
        )
        # if the build-in encoder has the attention results
        if 'att' in enc_results.keys():
            enc_outputs.update(
                att=enc_results['att']
            )
        # if the build-in encoder has the hidden results
        if 'hidden' in enc_results.keys():
            enc_outputs.update(
                hidden=enc_results['hidden']
            )
        return enc_outputs