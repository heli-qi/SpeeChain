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
from speechain.utilbox.import_util import import_class

from speechain.module.norm.feat_norm import FeatureNormalization
from speechain.module.augment.specaug import SpecAugment


class ASREncoder(Module):
    """

    """

    def module_init(self,
                    encoder: Dict,
                    frontend: Dict = None,
                    normalize: Dict or bool = None,
                    specaug: Dict or bool = None,
                    prenet: Dict = None):
        """

        Args:
            (mandatory) frontend:
            (optional) normalize:
            (optional) specaug:
            (optional) prenet:
            (mandatory) encoder:

        """
        # temporary register for connecting two sequential modules
        _prev_output_size = None

        # acoustic feature extraction frontend of the E2E ASR encoder
        if frontend is not None:
            frontend_class = import_class('speechain.module.' + frontend['type'])
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
            prenet_class = import_class('speechain.module.' + prenet['type'])
            prenet['conf'] = dict() if 'conf' not in prenet.keys() else prenet['conf']
            self.prenet = prenet_class(input_size=_prev_output_size, **prenet['conf'])
            _prev_output_size = self.prenet.output_size

        # main body of the E2E ASR encoder
        encoder_class = import_class('speechain.module.' + encoder['type'])
        encoder['conf'] = dict() if 'conf' not in encoder.keys() else encoder['conf']
        self.encoder = encoder_class(input_size=_prev_output_size, **encoder['conf'])
        self.output_size = self.encoder.output_size

    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor, epoch: int = None, domain: str = None):
        """

        Args:
            feat:
            feat_len:
            epoch:
            domain:

        Returns:

        """

        # acoustic feature extraction for the waveform input, do nothing for the feature input
        if feat.size(-1) == 1:
            assert hasattr(self, 'frontend'), \
                "Currently, we don't support time-domain ASR. Please specify a feature extraction frontend!"
            # no amp operations for the frontend calculation to make sure the feature accuracy
            with autocast(False):
                feat, feat_len = self.frontend(feat, feat_len)

        # feature normalization
        if hasattr(self, 'normalize'):
            feat, feat_len = self.normalize(feat, feat_len, epoch=epoch, group_ids=domain)

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

        feat, feat_mask, attmat, hidden = self.encoder(feat, feat_mask)
        return feat, feat_mask, attmat, hidden
