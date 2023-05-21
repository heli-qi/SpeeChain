"""
    Author: Sashi Novitasari
    Affiliation: NAIST (-2022)
    Date: 2022.08

    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.09
"""
import warnings

import torch
from torch.cuda.amp import autocast
from typing import Dict
from speechain.module.abs import Module
from speechain.utilbox.train_util import make_mask_from_len

from speechain.module.frontend.speech2linear import Speech2LinearSpec
from speechain.module.frontend.speech2mel import Speech2MelSpec
from speechain.module.norm.feat_norm import FeatureNormalization
from speechain.module.prenet.linear import LinearPrenet
from speechain.module.prenet.spk_embed import SpeakerEmbedPrenet
from speechain.module.transformer.decoder import TransformerDecoder
from speechain.module.postnet.conv1d import Conv1dPostnet


class ARTTSDecoder(Module):
    """
        Decoder Module for Autoregressive TTS model.
    """
    frontend_class_dict = dict(
        stft_spec=Speech2LinearSpec,
        mel_fbank=Speech2MelSpec
    )

    prenet_class_dict = dict(
        linear=LinearPrenet
    )

    decoder_class_dict = dict(
        transformer=TransformerDecoder
    )

    postnet_class_dict = dict(
        conv1d=Conv1dPostnet
    )

    def module_init(self,
                    decoder: Dict,
                    postnet: Dict,
                    frontend: Dict = None,
                    normalize: Dict or bool = True,
                    prenet: Dict = None,
                    spk_emb: Dict = None,
                    reduction_factor: int = 1):

        # temporary register for connecting two sequential modules
        _prev_output_size = None

        # --- Acoustic Feature Extraction Part --- #
        # acoustic feature extraction frontend of the E2E TTS decoder
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

        # reduction factor for acoustic feature sequence
        self.reduction_factor = reduction_factor
        _prev_output_size *= self.reduction_factor
        feat_dim = _prev_output_size

        # --- Main Body of TTS Decoder --- #
        # feature embedding layer of the E2E TTS decoder
        if prenet is not None:
            prenet_class = self.prenet_class_dict[prenet['type']]
            prenet['conf'] = dict() if 'conf' not in prenet.keys() else prenet['conf']
            self.prenet = prenet_class(input_size=_prev_output_size, **prenet['conf'])
            _prev_output_size = self.prenet.output_size

        # initialize speaker embedding layer
        if spk_emb is not None:
            # check the model dimension
            if 'd_model' in spk_emb.keys() and spk_emb['d_model'] != _prev_output_size:
                warnings.warn(f"Your input d_model ({spk_emb['d_model']}) is different from the one automatically "
                              f"generated during model initialization ({_prev_output_size})."
                              f"Currently, the automatically generated one is used.")

            spk_emb['d_model'] = _prev_output_size
            self.spk_emb = SpeakerEmbedPrenet(**spk_emb)

        # Initialize decoder
        decoder_class = self.decoder_class_dict[decoder['type']]
        decoder['conf'] = dict() if 'conf' not in decoder.keys() else decoder['conf']
        self.decoder = decoder_class(input_size=_prev_output_size, **decoder['conf'])
        _prev_output_size = self.decoder.output_size

        # initialize prediction layers (feature prediction & stop prediction)
        self.feat_pred = torch.nn.Linear(in_features=_prev_output_size, out_features=feat_dim)
        self.stop_pred = torch.nn.Linear(in_features=_prev_output_size, out_features=1)
        self.output_size = feat_dim

        # Initialize postnet of the decoder
        postnet_class = self.postnet_class_dict[postnet['type']]
        postnet['conf'] = dict() if 'conf' not in postnet.keys() else postnet['conf']
        self.postnet = postnet_class(input_size=feat_dim, **postnet['conf'])

    def forward(self,
                enc_text: torch.Tensor, enc_text_mask: torch.Tensor,
                feat: torch.Tensor, feat_len: torch.Tensor,
                spk_feat: torch.Tensor = None, spk_ids: torch.Tensor = None,
                epoch: int = None, is_test: bool = False):

        # --- Acoustic Feature Extraction Part --- #
        # in the training and validation stage, input data needs to be processed by the frontend
        if not is_test:
            # acoustic feature extraction for the raw waveform input
            if feat.size(-1) == 1:
                assert hasattr(self, 'frontend'), \
                    f"Currently, {self.__class__.__name__} doesn't support time-domain TTS. " \
                    f"Please specify a feature extraction frontend."
                # no amp operations for the frontend calculation to make sure the feature accuracy
                with autocast(False):
                    feat, feat_len = self.frontend(feat, feat_len)

            # feature normalization
            if hasattr(self, 'normalize'):
                feat, feat_len = self.normalize(feat, feat_len, group_ids=spk_ids, epoch=epoch)

            # acoustic feature length reduction
            if self.reduction_factor > 1:
                _residual = feat.size(1) % self.reduction_factor
                # clip the redundant part of acoustic features
                if _residual != 0:
                    feat = feat[:, :-_residual]

                # group the features by the reduction factor
                batch, feat_maxlen, feat_dim = feat.size()
                feat = feat.reshape(
                    batch, feat_maxlen // self.reduction_factor, feat_dim * self.reduction_factor
                )
                feat_len = torch.div(feat_len, self.reduction_factor, rounding_mode='floor').type(torch.long)

            # padding zeros at the beginning of acoustic feature sequence
            padded_feat = torch.nn.functional.pad(feat, (0, 0, 1, 0), "constant", 0)
            feat = padded_feat[:, :-1]
            # target feature & length, used for loss & metric calculation
            tgt_feat, tgt_feat_len = padded_feat[:, 1:], feat_len

        # in the testing stage, input data has already been processed and structured, so no frontend processing here
        else:
            tgt_feat, tgt_feat_len = None, None

        # --- Decoder Feature Transformation Part --- #
        # feature Embedding
        if hasattr(self, 'prenet'):
            feat, feat_len = self.prenet(feat, feat_len)

        # mask generation for the input text
        feat_mask = make_mask_from_len(feat_len)
        if feat.is_cuda:
            feat_mask = feat_mask.cuda(feat.device)

        # Speaker Embedding
        if hasattr(self, 'spk_emb'):
            # extract and process the speaker features (activation is not performed for random speaker feature)
            spk_feat_lookup, spk_feat = self.spk_emb(spk_ids=spk_ids, spk_feat=spk_feat)
            # combine the speaker features with the encoder outputs (and the decoder prenet outputs if specified)
            enc_text, feat = self.spk_emb.combine_spk_feat(spk_feat=spk_feat, spk_feat_lookup=spk_feat_lookup,
                                                           enc_output=enc_text, dec_input=feat)

        # Decoding
        dec_feat, self_attmat, encdec_attmat, hidden = self.decoder(src=enc_text, src_mask=enc_text_mask,
                                                                    tgt=feat, tgt_mask=feat_mask)
        pred_stop = self.stop_pred(dec_feat)
        pred_feat_before = self.feat_pred(dec_feat)
        pred_feat_after = pred_feat_before + self.postnet(pred_feat_before, feat_len)

        return pred_stop, pred_feat_before, pred_feat_after, \
            tgt_feat, tgt_feat_len, self_attmat, encdec_attmat, hidden

    def turn_on_dropout(self):
        """
        turn on the dropout layers of the decoder prenet during inference.
        Reference: the second paragraph from the bottom in Sec 2.2 of
            'Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions'
            https://arxiv.org/pdf/1712.05884.pdf

        """
        assert hasattr(self, 'prenet'), \
            "If you want to apply dropout during TTS inference, your TTS model should have a decoder prenet."
        if not self.prenet.training:
            self.prenet.train()
