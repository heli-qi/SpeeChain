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
from speechain.module.norm.feat_norm import FeatureNormalization
from speechain.module.prenet.spk_embed import SpeakerEmbedPrenet

from speechain.utilbox.train_util import make_len_from_mask, make_mask_from_len
from speechain.utilbox.import_util import import_class


class FastSpeech2Decoder(Module):
    """
        Decoder Module for Non-Autoregressive FastSpeech2 model.
    """

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
                    reduction_factor: int = 1):

        # reduction factor for acoustic feature sequence
        self.reduction_factor = reduction_factor

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
        duration_predictor_class = import_class('speechain.module.' + duration_predictor['type'])
        duration_predictor['conf'] = dict() if 'conf' not in duration_predictor.keys() else duration_predictor['conf']
        # the conv1d embedding layer of duration predictor is automatically turned off
        duration_predictor['conf']['use_conv_emb'] = False
        self.duration_predictor = duration_predictor_class(input_size=self.input_size, **duration_predictor['conf'])

        # pitch predictor initialization
        pitch_predictor_class = import_class('speechain.module.' + pitch_predictor['type'])
        pitch_predictor['conf'] = dict() if 'conf' not in pitch_predictor.keys() else pitch_predictor['conf']
        # the conv1d embedding layer of pitch predictor is automatically turned on
        pitch_predictor['conf']['use_conv_emb'] = True
        self.pitch_predictor = pitch_predictor_class(input_size=self.input_size, **pitch_predictor['conf'])

        # energy predictor initialization
        energy_predictor_class = import_class('speechain.module.' + energy_predictor['type'])
        energy_predictor['conf'] = dict() if 'conf' not in energy_predictor.keys() else energy_predictor['conf']
        # the conv1d embedding layer of energy predictor is automatically turned on
        energy_predictor['conf']['use_conv_emb'] = True
        self.energy_predictor = energy_predictor_class(input_size=self.input_size, **energy_predictor['conf'])

        # --- 3. Acoustic Feature, Energy, & Pitch Extraction Part --- #
        # acoustic feature extraction frontend of the E2E TTS decoder
        if feat_frontend is not None:
            feat_frontend_class = import_class('speechain.module.' + feat_frontend['type'])
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
        decoder_class = import_class('speechain.module.' + decoder['type'])
        decoder['conf'] = dict() if 'conf' not in decoder.keys() else decoder['conf']
        self.decoder = decoder_class(input_size=self.input_size, **decoder['conf'])

        # initialize prediction layers (feature prediction & stop prediction)
        self.feat_pred = torch.nn.Linear(in_features=self.decoder.output_size, out_features=feat_dim)

        # Initialize postnet of the decoder
        postnet_class = import_class('speechain.module.' + postnet['type'])
        postnet['conf'] = dict() if 'conf' not in postnet.keys() else postnet['conf']
        self.postnet = postnet_class(input_size=feat_dim, **postnet['conf'])

    @staticmethod
    def average_scalar_by_duration(frame_scalar: torch.Tensor, duration: torch.Tensor, duration_len: torch.Tensor):
        """
        Compute the average scalar value for each token in a batch of variable length frames.

        Args:
          frame_scalar:
            A float tensor of shape (batch_size, max_frame_len) containing the scalar values of each frame in the batch.
          duration:
            An int tensor of shape (batch_size, max_token_len) containing the duration of each token in the batch.
          duration_len:
            An int tensor of shape (batch_size,) containing the length of each sequence of tokens in the batch.

        Returns:
          A tuple containing two tensors:
          - token_scalar:
                A float tensor of shape (batch_size, max_token_len) containing the average scalar value of each token.
          - duration_len:
                An int tensor of shape (batch_size,) containing the length of each sequence of tokens in the batch.

        Note:
          The max_frame_len and max_token_len are not necessarily the same.
        """
        batch_size, max_frame_len = frame_scalar.size()
        max_token_len = duration.size(1)
        device = frame_scalar.device

        # Create indices tensor for gather operation
        end_frame_idxs = torch.cumsum(duration, dim=1).unsqueeze(-1)
        start_frame_idxs = torch.nn.functional.pad(end_frame_idxs[:, :-1], (0, 0, 1, 0), value=0)
        range_mat = torch.arange(max_frame_len, device=device)[None, None, :].expand(batch_size, max_token_len, -1)
        idxs = torch.where((range_mat >= start_frame_idxs) & (range_mat < end_frame_idxs), range_mat, -1)

        # Compute the mean scalar value for each token
        mask = idxs >= 0
        idxs = torch.clamp(idxs, min=0)

        # Gather the frame scalar values using the index tensor
        token_scalar = torch.gather(frame_scalar.unsqueeze(1).expand(-1, max_token_len, -1), 2, idxs)
        token_scalar = torch.where(mask, token_scalar, 0)
        token_scalar = token_scalar.sum(dim=2) / (mask.sum(dim=2) + 1e-10)

        return token_scalar, duration_len

    def proc_duration(self, duration: torch.Tensor,
                      min_frame_num: int = 0, max_frame_num: int = None, duration_alpha: torch.Tensor = None):
        # modify the duration by duration_alpha during evaluation
        if not self.training and duration_alpha is not None:
            duration = duration * duration_alpha

        # convert the negative numbers to zeros
        duration = torch.clamp(torch.round(duration), min=0)
        duration_zero_mask = duration == 0

        # round the phoneme duration to the nearest integer
        duration = torch.clamp(duration, min=round(min_frame_num / self.reduction_factor),
                    max=None if max_frame_num is None else round(max_frame_num / self.reduction_factor))
        duration = torch.where(duration_zero_mask, 0, duration)
        return duration

    def forward(self,
                enc_text: torch.Tensor, enc_text_mask: torch.Tensor,
                duration: torch.Tensor = None, duration_len: torch.Tensor = None,
                pitch: torch.Tensor = None, pitch_len: torch.Tensor = None,
                feat: torch.Tensor = None, feat_len: torch.Tensor = None,
                energy: torch.Tensor = None, energy_len: torch.Tensor = None,
                spk_feat: torch.Tensor = None, spk_ids: torch.Tensor = None,
                epoch: int = None, min_frame_num: int = None, max_frame_num: int = None,
                duration_alpha: torch.Tensor = None, energy_alpha: torch.Tensor = None,
                pitch_alpha: torch.Tensor = None):

        # --- 1. Speaker Embedding Combination --- #
        if hasattr(self, 'spk_emb'):
            # extract and process the speaker features (activation is not performed for random speaker feature)
            spk_feat_lookup, spk_feat = self.spk_emb(spk_ids=spk_ids, spk_feat=spk_feat)
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
        pred_duration_outputs = self.duration_predictor(enc_text, make_len_from_mask(enc_text_mask))
        # without duration gate predictions
        if len(pred_duration_outputs) == 2:
            pred_duration, enc_text_len = pred_duration_outputs
            pred_duration_gate = None
        # with duration gate predictions
        elif len(pred_duration_outputs) == 3:
            pred_duration, enc_text_len, pred_duration_gate = pred_duration_outputs
        else:
            raise RuntimeError

        # do teacher-forcing if ground-truth duration is given
        if duration is not None:
            # turn the duration from the second to the frame number
            used_duration = self.proc_duration(
                duration=duration / duration.sum(dim=-1, keepdims=True) * feat_len.unsqueeze(-1),
                min_frame_num=min_frame_num, max_frame_num=max_frame_num, duration_alpha=duration_alpha
            )
            used_duration_len = duration_len
        # do self-decoding by the predicted duration
        else:
            # mask the predicted duration by gate predictions if available
            if pred_duration_gate is not None:
                pred_duration[pred_duration_gate > 0] = 0.0
            # use the exp-converted predicted duration
            used_duration = self.proc_duration(
                duration=torch.exp(pred_duration) - 1,
                min_frame_num=min_frame_num, max_frame_num=max_frame_num, duration_alpha=duration_alpha
            )
            used_duration_len = enc_text_len
            used_duration.masked_fill_(~make_mask_from_len(enc_text_len, return_3d=False), 0.0)

        # --- 4. Pitch Prediction and Embedding --- #
        pred_pitch, enc_text_len = self.pitch_predictor(enc_text, enc_text_len)
        if pitch is not None:
            # turn the frame-wise pitch values into frame-averaged pitch values
            pitch, pitch_len = self.average_scalar_by_duration(pitch, used_duration, used_duration_len)
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
            energy, energy_len = self.average_scalar_by_duration(energy, used_duration, used_duration_len)
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
        # loop each sentence
        expand_enc_text_list = []
        for i in range(len(enc_text_len)):
            # calculate the number of frames needed for each token in the current sentence
            frame_counts = used_duration[i].long()
            # expand the phoneme embeddings by the number of frames for each token
            expand_enc_text_list.append(enc_text[i].repeat_interleave(frame_counts, dim=0))

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

        # the returned duration is used_duration (the actual one used for length regulation)
        return pred_feat_before, pred_feat_after, make_len_from_mask(dec_feat_mask), feat, feat_len,\
               pred_pitch, pitch, pitch_len, pred_energy, energy, energy_len,\
               pred_duration, pred_duration_gate, used_duration, used_duration_len, dec_attmat, dec_hidden
