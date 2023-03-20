"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2023.02
"""
import torch
import warnings

from torch.cuda.amp import autocast
from typing import Dict

from speechain.module.abs import Module
from speechain.module.frontend.speech2linear import Speech2LinearSpec
from speechain.module.frontend.speech2mel import Speech2MelSpec
from speechain.module.norm.feat_norm import FeatureNormalization
from speechain.module.prenet.spk_embed import SpeakerEmbedPrenet
from speechain.module.postnet.conv1d import Conv1dPostnet
from speechain.module.transformer.encoder import TransformerEncoder
from speechain.module.prenet.var_pred import Conv1dVarPredictor

from speechain.utilbox.train_util import make_len_from_mask, make_mask_from_len


class FastSpeech2Decoder(Module):
    """
        Decoder Module for Non-Autoregressive FastSpeech2 model.
    """
    feat_frontend_class_dict = dict(
        stft_spec=Speech2LinearSpec,
        mel_fbank=Speech2MelSpec
    )

    var_pred_class_dict = dict(
        conv1d=Conv1dVarPredictor
    )

    decoder_class_dict = dict(
        transformer=TransformerEncoder
    )

    postnet_class_dict = dict(
        conv1d=Conv1dPostnet
    )

    def module_init(self,
                    decoder: Dict,
                    postnet: Dict,
                    pitch_predictor: Dict,
                    energy_predictor: Dict,
                    duration_predictor: Dict,
                    spk_emb: Dict = None,
                    feat_frontend: Dict = None,
                    feat_normalize: Dict or bool = True,
                    pitch_normalize: Dict or bool = True,
                    energy_normalize: Dict or bool = True,
                    len_regulation_type: str = 'hard',
                    reduction_factor: int = 1,
                    min_frame_num: int = 0,
                    max_frame_num: int = None):

        # reduction factor for acoustic feature sequence
        self.reduction_factor = reduction_factor
        self.min_frame_num = min_frame_num
        self.max_frame_num = max_frame_num
        if len_regulation_type not in ['hard', 'soft']:
            raise ValueError(f"Unknown len_regulation_type {len_regulation_type}. It must be either 'hard' or 'soft'!")
        else:
            self.len_regulation_type = len_regulation_type

        # --- 1. Speaker Embedding Part --- #
        if spk_emb is not None:
            # check the model dimension
            if 'd_model' in spk_emb.keys() and spk_emb['d_model'] != self.input_size:
                warnings.warn(f"Your input d_model ({spk_emb['d_model']}) is different from the one automatically "
                              f"generated during model initialization ({self.input_size})."
                              f"Currently, the automatically generated one is used.")

            spk_emb['d_model'] = self.input_size
            self.spk_emb = SpeakerEmbedPrenet(**spk_emb)

        # --- 2. Variance Predictors Part --- #
        # duration predictor initialization
        duration_predictor_class = self.var_pred_class_dict[duration_predictor['type']]
        duration_predictor['conf'] = dict() if 'conf' not in duration_predictor.keys() else duration_predictor['conf']
        # the conv1d embedding layer of duration predictor is automatically turned off
        duration_predictor['conf']['use_conv_emb'] = False
        self.duration_predictor = duration_predictor_class(input_size=self.input_size, **duration_predictor['conf'])

        # pitch predictor initialization
        pitch_predictor_class = self.var_pred_class_dict[pitch_predictor['type']]
        pitch_predictor['conf'] = dict() if 'conf' not in pitch_predictor.keys() else pitch_predictor['conf']
        # the conv1d embedding layer of pitch predictor is automatically turned on
        pitch_predictor['conf']['use_conv_emb'] = True
        self.pitch_predictor = pitch_predictor_class(input_size=self.input_size, **pitch_predictor['conf'])

        # energy predictor initialization
        energy_predictor_class = self.var_pred_class_dict[energy_predictor['type']]
        energy_predictor['conf'] = dict() if 'conf' not in energy_predictor.keys() else energy_predictor['conf']
        # the conv1d embedding layer of energy predictor is automatically turned on
        energy_predictor['conf']['use_conv_emb'] = True
        self.energy_predictor = energy_predictor_class(input_size=self.input_size, **energy_predictor['conf'])

        # --- 3. Acoustic Feature, Energy, & Pitch Extraction Part --- #
        # acoustic feature extraction frontend of the E2E TTS decoder
        if feat_frontend is not None:
            feat_frontend_class = self.feat_frontend_class_dict[feat_frontend['type']]
            feat_frontend['conf'] = dict() if 'conf' not in feat_frontend.keys() else feat_frontend['conf']
            # feature frontend is automatically set to return frame-wise energy
            feat_frontend['conf']['return_energy'] = True
            self.feat_frontend = feat_frontend_class(**feat_frontend['conf'])
            feat_dim = self.feat_frontend.output_size * self.reduction_factor
        else:
            feat_dim = None

        # feature normalization layer
        if feat_normalize is not None and feat_normalize is not False:
            feat_normalize = dict() if feat_normalize is True else feat_normalize
            self.feat_normalize = FeatureNormalization(input_size=feat_dim, distributed=self.distributed,
                                                       **feat_normalize)
        # energy normalization layer
        if energy_normalize is not None and energy_normalize is not False:
            energy_normalize = dict() if energy_normalize is True else energy_normalize
            self.energy_normalize = FeatureNormalization(input_size=1, distributed=self.distributed,
                                                         **energy_normalize)

        # feature normalization layer
        if pitch_normalize is not None and pitch_normalize is not False:
            pitch_normalize = dict() if pitch_normalize is True else pitch_normalize
            self.pitch_normalize = FeatureNormalization(input_size=1, distributed=self.distributed,
                                                        **pitch_normalize)

        # --- 4. Mel-Spectrogram Decoder Part --- #
        # Initialize decoder
        decoder_class = self.decoder_class_dict[decoder['type']]
        decoder['conf'] = dict() if 'conf' not in decoder.keys() else decoder['conf']
        self.decoder = decoder_class(input_size=self.input_size, **decoder['conf'])

        # initialize prediction layers (feature prediction & stop prediction)
        self.feat_pred = torch.nn.Linear(in_features=self.decoder.output_size, out_features=feat_dim)

        # Initialize postnet of the decoder
        postnet_class = self.postnet_class_dict[postnet['type']]
        postnet['conf'] = dict() if 'conf' not in postnet.keys() else postnet['conf']
        self.postnet = postnet_class(input_size=feat_dim, **postnet['conf'])

    def average_scalar_by_duration(self, frame_scalar: torch.Tensor, frame_scalar_len: torch.Tensor,
                                   duration: torch.Tensor, duration_len: torch.Tensor):
        token_scalar = torch.zeros_like(duration)
        # loop each sentence
        for i in range(len(duration)):
            # for the hard length regulation (integer duration)
            if self.len_regulation_type != 'soft':
                _cursor = 0
                for j in range(duration_len[i]):
                    d = min(int(duration[i][j].item()), frame_scalar_len[i].item() - _cursor)
                    if d > 0:
                        token_scalar[i][j] += frame_scalar[i][_cursor: _cursor + d].mean()
                        _cursor += d
                    if _cursor >= frame_scalar_len[i]:
                        continue
            # for the soft length regulation (float duration)
            else:
                # the cursor represents the current position in frame_scalar[i]
                _cursor, _prev_weight = 0, 0
                # loop each token in the current sentence
                for j in range(duration_len[i]):
                    d = duration[i][j].item()
                    if d > 0:
                        # for the phoneme that doesn't cross two or more frames
                        if d < _prev_weight:
                            token_scalar[i][j] += frame_scalar[i][_cursor - 1]
                            _prev_weight = _prev_weight - d
                            # cursor is not updated in this situation
                        # for the phoneme that crosses two or more frames
                        else:
                            # integer part of the soft duration, represents the current frames
                            int_d = int(d - _prev_weight)
                            # fraction part of the soft duration, represents the next frame
                            frac_d = d - _prev_weight - int_d
                            # check the boundary condition for the cursor
                            if int_d >= frame_scalar_len[i].item() - _cursor:
                                int_d, frac_d = frame_scalar_len[i].item() - _cursor, 0
                            # the duration of the previous frame
                            if _prev_weight > 0:
                                token_scalar[i][j] += _prev_weight * frame_scalar[i][_cursor - 1]
                            # the duration of the integer part (the current frames)
                            if int_d > 0:
                                token_scalar[i][j] += frame_scalar[i][_cursor: _cursor + int_d].sum()
                            # the duration of the fraction part (the next frame)
                            if frac_d > 0:
                                token_scalar[i][j] += frac_d * frame_scalar[i][_cursor + int_d]
                            # take the average scalar by the three weights
                            token_scalar[i][j] /= (_prev_weight + int_d + frac_d)
                            # update the cursor and watch the end condition
                            _cursor, _prev_weight = _cursor + int_d + 1, 1 - frac_d
                            # check the end condition for the cursor
                            if _cursor >= frame_scalar_len[i] + 1:
                                continue
        return token_scalar, duration_len

    def proc_duration(self, duration: torch.Tensor, duration_alpha: torch.Tensor = None):
        # modify the duration by duration_alpha during evaluation
        if not self.training and duration_alpha is not None:
            duration = duration * duration_alpha

        # round the phoneme duration to the nearest integer in the hard mode
        if self.len_regulation_type == 'hard':
            duration = torch.clamp(torch.round(duration), min=round(self.min_frame_num / self.reduction_factor),
                        max=None if self.max_frame_num is None else round(self.max_frame_num / self.reduction_factor))
        # keep the phoneme duration to be the float numbers in the soft mode
        elif self.len_regulation_type == 'soft':
            duration = torch.clamp(duration, min=self.min_frame_num / self.reduction_factor,
                        max=None if self.max_frame_num is None else self.max_frame_num / self.reduction_factor)
        else:
            raise RuntimeError(f"Unknown len_regulation_type {self.len_regulation_type}."
                               "It must be either 'hard' or 'soft'!")
        return duration

    def forward(self,
                enc_text: torch.Tensor, enc_text_mask: torch.Tensor,
                duration: torch.Tensor = None, duration_len: torch.Tensor = None,
                pitch: torch.Tensor = None, pitch_len: torch.Tensor = None,
                feat: torch.Tensor = None, feat_len: torch.Tensor = None,
                energy: torch.Tensor = None, energy_len: torch.Tensor = None,
                spk_feat: torch.Tensor = None, spk_ids: torch.Tensor = None,
                epoch: int = None, rand_spk_feat: bool = False,
                min_f2t_ratio: float = None, max_f2t_ratio: float = None,
                duration_alpha: torch.Tensor = None, energy_alpha: torch.Tensor = None,
                pitch_alpha: torch.Tensor = None):

        # --- 1. Speaker Embedding Combination --- #
        if hasattr(self, 'spk_emb'):
            # extract and process the speaker features (activation is not performed for random speaker feature)
            spk_feat_lookup, spk_feat = self.spk_emb(spk_ids=spk_ids, spk_feat=spk_feat, spk_feat_act=not rand_spk_feat)
            # combine the speaker features with the encoder outputs (and the decoder prenet outputs if specified)
            enc_text, _ = self.spk_emb.combine_spk_feat(spk_feat=spk_feat, spk_feat_lookup=spk_feat_lookup,
                                                        enc_output=enc_text)

        # --- 2. Acoustic Feature, Pitch, Energy Extraction --- #
        # in the training and validation stage, input feature data needs to be processed by the feature frontend
        if feat is not None:
            # acoustic feature extraction for the raw waveform input
            if feat.size(-1) == 1:
                assert hasattr(self, 'feat_frontend'), \
                    f"Currently, {self.__class__.__name__} doesn't support time-domain TTS. " \
                    f"Please specify a feature extraction frontend."
                # no amp operations for the frontend calculation to make sure the feature extraction accuracy
                with autocast(False):
                    feat, feat_len, energy, energy_len = self.feat_frontend(feat, feat_len)

            # feature normalization
            if hasattr(self, 'feat_normalize'):
                feat, feat_len = self.feat_normalize(feat, feat_len, group_ids=spk_ids, epoch=epoch)

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

        if pitch is not None:
            # pitch normalization
            if hasattr(self, 'pitch_normalize'):
                pitch, pitch_len = self.pitch_normalize(pitch, pitch_len, group_ids=spk_ids, epoch=epoch)

            # pitch length reduction
            if self.reduction_factor > 1:
                _residual = pitch.size(1) % self.reduction_factor
                # clip the redundant part of pitch
                if _residual != 0:
                    pitch = pitch[:, :-_residual]

                # average the pitch by the reduction factor
                batch, pitch_maxlen = pitch.size()
                pitch = pitch.reshape(
                    batch, pitch_maxlen // self.reduction_factor, self.reduction_factor
                ).mean(dim=-1)
                pitch_len = torch.div(pitch_len, self.reduction_factor, rounding_mode='floor').type(torch.long)

        if energy is not None:
            # energy normalization
            if hasattr(self, 'energy_normalize'):
                energy, energy_len = self.energy_normalize(energy, energy_len, group_ids=spk_ids, epoch=epoch)

            # energy length reduction
            if self.reduction_factor > 1:
                _residual = energy.size(1) % self.reduction_factor
                # clip the redundant part of energy
                if _residual != 0:
                    energy = energy[:, :-_residual]

                # average the energy by the reduction factor
                batch, energy_maxlen = energy.size()
                energy = energy.reshape(
                    batch, energy_maxlen // self.reduction_factor, self.reduction_factor
                ).mean(dim=-1)
                energy_len = torch.div(energy_len, self.reduction_factor, rounding_mode='floor').type(torch.long)

        # target labels checking for training stage
        if self.training:
            assert energy is not None and energy_len is not None and \
                   pitch is not None and pitch_len is not None and \
                   duration is not None and duration_len is not None, \
                "During training, please give the ground-truth of energy, pitch, and duration."

        # --- 3. Duration Prediction --- #
        # note: pred_duration here is in the log domain!
        pred_duration, enc_text_len = self.duration_predictor(enc_text, make_len_from_mask(enc_text_mask))
        # do teacher-forcing if ground-truth duration is given
        if duration is not None:
            # turn the duration from the second to the frame number
            used_duration = self.proc_duration(
                duration / duration.sum(dim=-1, keepdims=True) * feat_len.unsqueeze(-1), duration_alpha=duration_alpha)
            used_duration_len = duration_len
        # do self-decoding by the predicted duration
        else:
            # use the exp-converted predicted duration
            used_duration = self.proc_duration(torch.exp(pred_duration) - 1, duration_alpha=duration_alpha)
            used_duration_len = enc_text_len
            used_duration.masked_fill_(~make_mask_from_len(enc_text_len, return_3d=False), 0.0)

        # --- 4. Pitch Prediction and Embedding --- #
        pred_pitch, enc_text_len = self.pitch_predictor(enc_text, enc_text_len)
        if pitch is not None:
            # turn the frame-wise pitch values into frame-averaged pitch values
            pitch, pitch_len = self.average_scalar_by_duration(pitch, pitch_len, used_duration, used_duration_len)
        # during training, ground-truth pitch is used for embedding
        # during validation & evaluation, predicted pitch is used for embedding
        used_pitch = pitch if self.training else pred_pitch
        # modify the pitch by pitch_alpha during evaluation
        if not self.training and pitch_alpha is not None:
            used_pitch = used_pitch * pitch_alpha
        emb_pitch = self.pitch_predictor.emb_pred_scalar(used_pitch)

        # --- 5. Energy Prediction and Embedding --- #
        pred_energy, enc_text_len = self.energy_predictor(enc_text, enc_text_len)
        if energy is not None:
            # turn the frame-wise energy values into frame-averaged energy values
            energy, energy_len = self.average_scalar_by_duration(energy, energy_len, used_duration, used_duration_len)
        # during training, ground-truth energy is used for embedding
        # during validation & evaluation, predicted energy is used for embedding
        used_energy = energy if self.training else pred_energy
        # modify the energy by energy_alpha during evaluation
        if not self.training and energy_alpha is not None:
            used_energy = used_energy * energy_alpha
        emb_energy = self.energy_predictor.emb_pred_scalar(used_energy)

        # --- 6. Pitch & Energy Embedding Combination --- #
        enc_text = enc_text + emb_pitch + emb_energy

        # --- 7. Length Regulation --- #
        # produce the flags which indicate whether to skip the utterance or not
        skip_flags = torch.zeros(enc_text_len.size(0), dtype=torch.bool, device=enc_text_len.device)
        if min_f2t_ratio is not None:
            skip_flags |= (used_duration.sum(-1) / enc_text_len) < min_f2t_ratio
        if max_f2t_ratio is not None:
            skip_flags |= (used_duration.sum(-1) / enc_text_len) > max_f2t_ratio

        # loop each sentence
        expand_enc_text_list = []
        for i in range(len(enc_text_len)):
            if skip_flags[i]:
                continue
            # used to record the current position in _expand_enc_text for proper length
            _expand_enc_text = []
            # for the hard length regulation (integer duration)
            if self.len_regulation_type != 'soft':
                # loop each token in the current sentence
                for j in range(enc_text_len[i]):
                    d = int(used_duration[i][j].item())
                    if d > 0:
                        # expand the current phoneme embedding by the phoneme duration
                        _expand_enc_text.append(enc_text[i][j].unsqueeze(0).expand(d, -1))
            # for the soft length regulation (float duration)
            else:
                _prev_weight = 0
                # loop each token in the current sentence
                for j in range(enc_text_len[i]):
                    d = used_duration[i][j].item()
                    if d > 0:
                        # for the phoneme that doesn't cross two or more frames
                        if d < _prev_weight:
                            _expand_enc_text[-1] = _expand_enc_text[-1] + d * enc_text[i][j].unsqueeze(0)
                            _prev_weight = _prev_weight - d
                        # for the phoneme that crosses two or more frames
                        else:
                            # integer part of the soft duration, represents the current frames
                            int_d = int(d - _prev_weight)
                            # fraction part of the soft duration, represents the next frame
                            frac_d = d - _prev_weight - int_d
                            # the weight for the previous frame
                            if _prev_weight > 0:
                                # += here will cause RuntimeError, so the operation should be out-place
                                _expand_enc_text[-1] = _expand_enc_text[-1] + _prev_weight * enc_text[i][j].unsqueeze(0)
                            # expand the phoneme embedding by the integer part of duration
                            if int_d > 0:
                                _expand_enc_text.append(enc_text[i][j].unsqueeze(0).expand(int_d, -1))
                            # the weight for the next frame (skip the last token to avoid an incomplete frame)
                            if frac_d > 0 and j < enc_text_len[i] - 1:
                                _expand_enc_text.append(frac_d * enc_text[i][j].unsqueeze(0))
                            # update the cursor to the second next frame
                            _prev_weight = 1 - frac_d

            # register the expanded enc_text for the current sentence
            expand_enc_text_list.append(torch.cat(_expand_enc_text))

        # teacher-forcing by feat_len if it is given
        if feat_len is not None:
            expand_enc_text_len = feat_len
        # self-decoding by expand_enc_text_list if feat_len is not given
        else:
            expand_enc_text_len = torch.LongTensor([len(i) for i in expand_enc_text_list]).to(enc_text.device)
        expand_enc_text_maxlen = expand_enc_text_len.max().item()

        # assemble all the expanded enc_text from the list into a single 3d tensor
        for i in range(len(expand_enc_text_list)):
            len_diff = expand_enc_text_maxlen - len(expand_enc_text_list[i])
            if len_diff > 0:
                expand_enc_text_list[i] = torch.nn.functional.pad(expand_enc_text_list[i], (0, 0, 0, len_diff), "constant", 0)
            elif len_diff < 0:
                # note: len_diff is a negative integer
                expand_enc_text_list[i] = expand_enc_text_list[i][:len_diff].contiguous()
            expand_enc_text_list[i] = expand_enc_text_list[i].unsqueeze(0)
        expand_enc_text = torch.cat(expand_enc_text_list)

        # --- 6. Mel-Spectrogram Prediction --- #
        dec_feat, dec_feat_mask, dec_attmat, dec_hidden = self.decoder(expand_enc_text, make_mask_from_len(expand_enc_text_len))
        pred_feat_before = self.feat_pred(dec_feat)
        pred_feat_after = pred_feat_before + self.postnet(pred_feat_before, make_len_from_mask(dec_feat_mask))

        # delete the utterances that will be skipped
        pred_pitch, pred_energy = pred_pitch[~skip_flags], pred_energy[~skip_flags]
        pred_duration, used_duration, used_duration_len = \
            pred_duration[~skip_flags], used_duration[~skip_flags], used_duration_len[~skip_flags]
        if pitch is not None:
            pitch, pitch_len = pitch[~skip_flags], pitch_len[~skip_flags]
        if energy is not None:
            energy, energy_len = energy[~skip_flags], energy_len[~skip_flags]

        # the returned duration is used_duration (the actual one used for length regulation)
        return pred_feat_before, pred_feat_after, make_len_from_mask(dec_feat_mask), skip_flags, feat, feat_len,\
               pred_pitch, pitch, pitch_len, pred_energy, energy, energy_len,\
               pred_duration, used_duration, used_duration_len, dec_attmat, dec_hidden
