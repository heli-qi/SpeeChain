"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch
from torch.cuda.amp import autocast
from typing import Dict
from speechain.module.abs import Module
from speechain.utilbox.train_util import make_mask_from_len

from speechain.module.frontend.speech2linear import Speech2LinearSpec
from speechain.module.frontend.speech2mel import Speech2MelSpec
from speechain.module.norm.feat_norm import FeatureNormalization
from speechain.module.augment.specaug import SpecAugment
from speechain.module.prenet.conv2d import Conv2dPrenet
from speechain.module.transformer.encoder import TransformerEncoder


class ASREncoder(Module):
    """

    """
    frontend_class_dict = dict(
        speech2linear=Speech2LinearSpec,
        speech2mel=Speech2MelSpec
    )

    prenet_class_dict = dict(
        conv2d=Conv2dPrenet
    )

    encoder_class_dict = dict(
        transformer=TransformerEncoder
    )

    def module_init(self,
                    encoder: Dict,
                    frontend: Dict = None,
                    normalize: Dict or bool = None,
                    specaug: Dict or bool = None,
                    prenet: Dict = None):
        """

        Args:
            frontend:
            normalize:
            specaug:
            encoder:
            prenet:

        """
        # temporary register for connecting two sequential modules
        _prev_output_size = None

        # acoustic feature extraction frontend of the E2E ASR encoder
        if frontend is not None:
            frontend_class = self.frontend_class_dict[frontend['type']]
            frontend['conf'] = dict() if 'conf' not in frontend.keys() else frontend['conf']
            self.frontend = frontend_class(**frontend['conf'])
            _prev_output_size = self.frontend.output_size

        # feature normalization layer
        if normalize is not None and normalize is not False:
            normalize = dict() if normalize is True else normalize
            self.normalize = FeatureNormalization(input_size=_prev_output_size, distributed=self.distributed,
                                                  **normalize)
            _prev_output_size = self.normalize.output_size

        # SpecAugment layer
        if specaug is not None and specaug is not False:
            specaug = dict() if specaug is True else specaug
            self.specaug = SpecAugment(input_size=_prev_output_size,
                                       feat_norm=normalize is not None and normalize is not False, **specaug)
            _prev_output_size = self.specaug.output_size

        # feature embedding layer of the E2E ASR encoder
        if prenet is not None:
            prenet_class = self.prenet_class_dict[prenet['type']]
            prenet['conf'] = dict() if 'conf' not in prenet.keys() else prenet['conf']
            self.prenet = prenet_class(input_size=_prev_output_size, **prenet['conf'])
            _prev_output_size = self.prenet.output_size

        # main body of the E2E ASR encoder
        encoder_class = self.encoder_class_dict[encoder['type']]
        encoder['conf'] = dict() if 'conf' not in encoder.keys() else encoder['conf']
        self.encoder = encoder_class(input_size=_prev_output_size, **encoder['conf'])


    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor, spk_ids: torch.Tensor = None, epoch: int = None):
        """

        Args:
            feat:
            feat_len:
            spk_ids:

        Returns:

        """

        # acoustic feature extraction
        if hasattr(self, 'frontend'):
            # no amp operations for the frontend calculation to make sure the feature accuracy
            with autocast(False):
                feat, feat_len = self.frontend(feat, feat_len)

        # feature normalization
        if hasattr(self, 'normalize'):
            feat, feat_len = self.normalize(feat, feat_len, group_ids=spk_ids, epoch=epoch)

        # SpecAugment, only activated during training
        if self.training and hasattr(self, "specaug"):
            feat, feat_len = self.specaug(feat, feat_len)

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