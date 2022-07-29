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

class ASREncoder(Module):
    """

    """
    def module_init(self,
                    encoder: Dict,
                    frontend: Dict = None,
                    prenet: Dict = None):
        """

        Args:
            frontend:
            encoder:
            prenet:

        Returns:

        """
        _prev_output_size = None

        if frontend is not None:
            frontend_class = import_class('speechain.module.' + frontend['type'])
            frontend['conf'] = dict() if 'conf' not in frontend.keys() else frontend['conf']
            self.frontend = frontend_class(**frontend['conf'])
            _prev_output_size = self.frontend.output_size

        if prenet is not None:
            prenet_class = import_class('speechain.module.' + prenet['type'])
            prenet['conf'] = dict() if 'conf' not in prenet.keys() else prenet['conf']
            self.prenet = prenet_class(input_size=_prev_output_size, **prenet['conf'])
            _prev_output_size = self.prenet.output_size

        encoder_class = import_class('speechain.module.' + encoder['type'])
        encoder['conf'] = dict() if 'conf' not in encoder.keys() else encoder['conf']
        self.encoder = encoder_class(input_size=_prev_output_size, **encoder['conf'])


    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor):
        """

        Args:
            feat:
            feat_len:

        Returns:

        """

        # acoustic feature extraction
        if hasattr(self, 'frontend'):
            # no amp operations for the frontend calculation to make sure the feature accuracy
            with autocast(False):
                feat, feat_len = self.frontend(feat, feat_len)

        # feature embedding by the encoder prenet
        if hasattr(self, 'prenet'):
            feat, feat_len = self.prenet(feat, feat_len)

        # mask generation for the embedded features
        feat_mask = make_mask_from_len(feat_len)
        if feat.is_cuda:
            feat_mask = feat_mask.cuda(feat.device)

        enc_results = self.encoder(feat, feat_mask)

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