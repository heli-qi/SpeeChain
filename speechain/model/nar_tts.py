"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2023.02
"""
import copy

import torch
import warnings
import numpy as np

from typing import Dict, List

import torchaudio

from speechain.tokenizer.char import CharTokenizer
from speechain.tokenizer.g2p import GraphemeToPhonemeTokenizer
from speechain.model.abs import Model

from speechain.module.encoder.tts import TTSEncoder
from speechain.module.decoder.nar_tts import FastSpeech2Decoder

from speechain.criterion.least_error import LeastError

from speechain.utilbox.data_loading_util import parse_path_args
from speechain.utilbox.tensor_util import to_cpu
from speechain.utilbox.train_util import get_min_indices_by_freq
from speechain.utilbox.sb_util import get_speechbrain_hifigan


class FastSpeech2(Model):
    """
        NonAuto-Regressive FastSpeech2 Text-To-Speech Synthesis Model. (single-speaker & multi-speaker)

    """

    def module_init(self,
                    token_type: str,
                    token_path: str,
                    enc_emb: Dict,
                    encoder: Dict,
                    pitch_predictor: Dict,
                    energy_predictor: Dict,
                    duration_predictor: Dict,
                    feat_frontend: Dict,
                    decoder: Dict,
                    enc_prenet: Dict = None,
                    dec_postnet: Dict = None,
                    feat_normalize: Dict or bool = True,
                    pitch_normalize: Dict or bool = True,
                    energy_normalize: Dict or bool = True,
                    spk_list: str = None,
                    spk_emb: Dict = None,
                    sample_rate: int = 22050,
                    vocoder: str = 'hifigan',
                    audio_format: str = 'wav',
                    reduction_factor: int = 1,
                    return_att_type: List[str] or str = None,
                    return_att_head_num: int = 2,
                    return_att_layer_num: int = 2):

        # --- 1. Model-Customized Part Initialization --- #
        # initialize the tokenizer
        if token_type == 'char':
            self.tokenizer = CharTokenizer(token_path, copy_path=self.result_path)
        elif token_type == 'mfa':
            self.tokenizer = GraphemeToPhonemeTokenizer(token_path, copy_path=self.result_path)
        else:
            raise ValueError(f"Unknown token type {token_type}. "
                             f"Currently, {self.__class__.__name__} supports one of ['char', 'mfa'].")

        # initialize the speaker list if given
        if spk_list is not None:
            if not isinstance(spk_list, list):
                spk_list = [spk_list]
            spk_list = np.concatenate([np.loadtxt(parse_path_args(s_l), dtype=str) for s_l in spk_list], axis=0)
            # when the input file is idx2spk, only retain the column of speaker ids
            if len(spk_list.shape) == 2:
                assert spk_list.shape[1] == 2
                spk_list = spk_list[:, 1]
            # otherwise, the input file must be spk_list which is a single-column file and each row is a speaker id
            elif len(spk_list.shape) != 1:
                raise RuntimeError
            # 1. remove redundant elements; 2. sort up the speaker ids in order
            spk_list = sorted(set(spk_list))
            # 3. get the corresponding indices (start from 1 since 0 is reserved for unknown speakers)
            self.idx2spk = dict(zip(range(1, len(spk_list) + 1), spk_list))
            # 4. exchange the positions of indices and speaker ids
            self.spk2idx = dict(map(reversed, self.idx2spk.items()))

        # initialize the sampling rate, mainly used for visualizing the input audio during training
        self.sample_rate = sample_rate
        self.audio_format = audio_format.lower()
        self.reduction_factor = reduction_factor

        if return_att_type is None:
            self.return_att_type = ['enc', 'dec']
        else:
            self.return_att_type = return_att_type if isinstance(return_att_type, List) else [return_att_type]
        for i in range(len(self.return_att_type)):
            if self.return_att_type[i].lower() in ['enc', 'dec']:
                self.return_att_type[i] = self.return_att_type[i].lower()
            else:
                raise ValueError("The elements of your input return_att_type must be one of ['enc', 'dec'], "
                                 f"but got {self.return_att_type[i]}!")
        self.return_att_head_num = return_att_head_num
        self.return_att_layer_num = return_att_layer_num

        # --- 2. Module Part Construction --- #
        # --- 2.1. Encoder construction --- #
        # the vocabulary size is given by the built-in tokenizer instead of the input configuration
        if 'vocab_size' in enc_emb['conf'].keys():
            if enc_emb['conf']['vocab_size'] != self.tokenizer.vocab_size:
                warnings.warn(f"Your input vocabulary size is different from the one obtained from the built-in "
                              f"tokenizer ({self.tokenizer.vocab_size}). The latter one will be used to initialize the "
                              f"encoder for correctness.")
            enc_emb['conf'].pop('vocab_size')
        self.encoder = TTSEncoder(
            vocab_size=self.tokenizer.vocab_size,
            embedding=enc_emb,
            prenet=enc_prenet,
            encoder=encoder
        )

        # --- 2.2. Decoder construction --- #
        # check the sampling rate of the decoder frontend
        if 'sr' not in feat_frontend['conf'].keys():
            # update the sampling rate of the frontend by the built-in one in the model
            feat_frontend['conf']['sr'] = self.sample_rate
        elif feat_frontend['conf']['sr'] != self.sample_rate:
            raise RuntimeError(f"The sampling rate given in your feat_frontend['conf'] ({feat_frontend['conf']['sr']}) "
                               f"is different from your given sample_rate ({self.sample_rate})!")

        # check the speaker embedding configuration
        if spk_emb is not None:
            # speaker number for the close-set multi-speaker TTS
            if hasattr(self, 'spk2idx'):
                if 'spk_num' in spk_emb.keys() and spk_emb['spk_num'] != len(self.spk2idx) + 1:
                    warnings.warn("Your input spk_num is different from the number of speakers in your given spk_list. "
                                  f"Currently, the spk_num is set to {len(self.spk2idx) + 1}.")
                # all seen speakers plus an unknown speaker (ID: 0)
                spk_emb['spk_num'], spk_emb['use_lookup'] = len(self.spk2idx) + 1, True
            elif 'use_lookup' in spk_emb.keys() and spk_emb['use_lookup']:
                raise RuntimeError("Please give spk_list in model['customize_conf'] if you want to use speaker lookup "
                                   "table for close-set multi-speaker TTS.")

        self.decoder = FastSpeech2Decoder(
            input_size=self.encoder.output_size,
            spk_emb=spk_emb,
            feat_frontend=feat_frontend,
            feat_normalize=feat_normalize,
            pitch_normalize=pitch_normalize,
            energy_normalize=energy_normalize,
            pitch_predictor=pitch_predictor,
            energy_predictor=energy_predictor,
            duration_predictor=duration_predictor,
            decoder=decoder,
            postnet=dec_postnet,
            distributed=self.distributed,
            reduction_factor=self.reduction_factor
        )

        # --- 3. Vocoding Backend Construction --- #
        assert vocoder in ['gl', 'hifigan'], \
            f"Currently, we only support 'gl' and 'hifigan' as vocoder, but got vocoder={vocoder}!"
        self.vocoder = vocoder
        if self.vocoder == 'gl':
            self.vocode_func = self.decoder.feat_frontend.recover
        else:
            self.vocode_func = get_speechbrain_hifigan(device=self.device, sample_rate=self.sample_rate,
                                                       use_multi_speaker=spk_emb is not None)

    @staticmethod
    def bad_cases_selection_init_fn() -> List[List[str or int]] or None:
        return [
            ['feat_token_len_ratio', 'max', 30],
            ['feat_token_len_ratio', 'min', 30],
            ['feat_len', 'max', 30],
            ['feat_len', 'min', 30],
            ['duration_zero_f1', 'min', 30]
        ]

    def criterion_init(self,
                       feat_loss: Dict = None,
                       pitch_loss: Dict = None,
                       energy_loss: Dict = None,
                       duration_loss: Dict = None):

        # acoustic feature loss, default to be MAE
        if feat_loss is None:
            feat_loss = dict(loss_type='L1')
        self.feat_loss = LeastError(**feat_loss)

        # pitch loss, default to be MSE
        if pitch_loss is None:
            pitch_loss = dict(loss_type='L2')
        self.pitch_loss = LeastError(**pitch_loss)

        # energy loss, default to be MSE
        if energy_loss is None:
            energy_loss = dict(loss_type='L2')
        self.energy_loss = LeastError(**energy_loss)

        # phoneme duration loss, default to be MSE
        if duration_loss is None:
            duration_loss = dict(loss_type='L2')
        self.duration_loss = LeastError(**duration_loss)

    def module_forward(self,
                       epoch: int = None,
                       text: torch.Tensor = None,
                       text_len: torch.Tensor = None,
                       duration: torch.Tensor = None,
                       duration_len: torch.Tensor = None,
                       pitch: torch.Tensor = None,
                       pitch_len: torch.Tensor = None,
                       feat: torch.Tensor = None,
                       feat_len: torch.Tensor = None,
                       energy: torch.Tensor = None,
                       energy_len: torch.Tensor = None,
                       spk_feat: torch.Tensor = None,
                       spk_ids: torch.Tensor = None,
                       rand_spk_feat: bool = False,
                       return_att: bool = False,
                       min_frame_num: int = 0,
                       max_frame_num: int = None,
                       duration_alpha: torch.Tensor = None,
                       energy_alpha: torch.Tensor = None,
                       pitch_alpha: torch.Tensor = None,
                       **kwargs) -> Dict:
        """

        Args:
            feat: (batch, feat_maxlen, feat_dim)
                The input speech data (grouped or downsampled and edge-padded).
            feat_len: (batch,)
                The lengths of input speech data
            text: (batch, text_maxlen)
                The input text data with <sos/eos> at the beginning and end
            text_len: (batch,)
                The lengths of input text data
            duration: (batch, text_maxlen)
                The duration data for each token in text.
            duration_len: (batch,)
                The lengths of input duration data
            pitch: (batch, text_maxlen)
                The pitch data for each token in text.
            pitch_len: (batch,)
                The lengths of input pitch data
            energy: (batch, text_maxlen)
                The energy data for each token in text.
            energy_len: (batch,)
                The lengths of input energy data
            spk_feat: (batch, 1, speaker embedding dim)
                Pre-extracted speaker embedding. (None means single-speaker TTS)
            spk_ids: (batch,)
                The speaker ids of each speech data. In the form of integer values.
            rand_spk_feat: bool
                Whether spk_feat is randomly generated.
            epoch: int
                The number of the current training epoch.
                Mainly used for mean&std calculation in the feature normalization
            return_att: bool
                Controls whether the attention matrices of each layer in the encoder and decoder will be returned.
            # Arguments for controllable TTS received from self.inference()
            duration_alpha:
            energy_alpha:
            pitch_alpha:
            kwargs:
                Temporary register used to store the redundant arguments.

        Returns:
            A dictionary containing all the TTS model outputs (feature, eos bernouli prediction) necessary to calculate the losses

        """
        # text checking
        assert text is not None and text_len is not None
        assert text_len.size(0) == text.size(0), \
            "The amounts of sentences and their lengths are not equal to each other."
        # feat checking
        if feat is not None and feat_len is not None:
            assert feat.size(0) == text.size(0) and feat_len.size(0) == text_len.size(0), \
                "The amounts of feat and text are not equal to each other."
            assert feat_len.size(0) == feat.size(0), \
                "The amounts of feat and their lengths are not equal to each other."
        elif (feat is None) ^ (feat_len is None):
            raise RuntimeError(f"In {self.__class__.__name__}, "
                               f"feat and feat_len must be None or not None at the same time! "
                               f"But got feat={feat} and feat_len={feat_len}.")
        # pitch checking
        if pitch is not None and pitch_len is not None:
            assert pitch.size(0) == feat.size(0) and pitch_len.size(0) == feat_len.size(0), \
                "The amounts of pitch and feat are not equal to each other."
            assert pitch_len.size(0) == pitch.size(0), \
                "The amounts of pitch and their lengths are not equal to each other."
        elif (pitch is None) ^ (pitch_len is None):
            raise RuntimeError(f"In {self.__class__.__name__}, "
                               f"pitch and pitch_len must be None or not None at the same time! "
                               f"But got pitch={pitch} and pitch_len={pitch_len}.")
        # energy checking
        if energy is not None and energy_len is not None:
            assert energy.size(0) == feat.size(0) and energy_len.size(0) == feat_len.size(0), \
                "The amounts of energy and feat are not equal to each other."
            assert energy_len.size(0) == energy.size(0), \
                "The amounts of energy and their lengths are not equal to each other."
        elif (energy is None) ^ (energy_len is None):
            raise RuntimeError(f"In {self.__class__.__name__}, "
                               f"energy and energy_len must be None or not None at the same time! "
                               f"But got energy={energy} and energy_len={energy_len}.")

        # text preprocessing before duration checking
        # remove the <sos/eos> at the beginning and the end of each sentence
        for i in range(text_len.size(0)):
            text[i, text_len[i] - 1] = self.tokenizer.ignore_idx
        text, text_len = text[:, 1:-1], text_len - 2
        
        # duration checking
        if duration is not None and duration_len is not None:
            assert duration.size(0) == text.size(0), \
                "The amounts of durations and text are not equal to each other."
            assert duration_len.size(0) == text_len.size(0), \
                "The amounts of durations and text lengths are not equal to each other."
            # check the length of duration and text
            assert False not in [len(text[i]) == len(duration[i]) for i in range(len(text))], \
                "The lengths of individual duration and text data don't match with each other."
        elif (duration is None) ^ (duration_len is None):
            raise RuntimeError(f"In {self.__class__.__name__}, "
                               f"duration and duration_len must be None or not None at the same time! "
                               f"But got duration={duration} and duration_len={duration_len}.")

        # Encoding the text data
        enc_text, enc_text_mask, enc_attmat, enc_hidden = self.encoder(text=text, text_len=text_len)

        # Decoding
        pred_feat_before, pred_feat_after, pred_feat_len, tgt_feat, tgt_feat_len, \
        pred_pitch, tgt_pitch, tgt_pitch_len, \
        pred_energy, tgt_energy, tgt_energy_len, \
        pred_duration, tgt_duration, tgt_duration_len, \
        dec_attmat, dec_hidden = self.decoder(enc_text=enc_text, enc_text_mask=enc_text_mask,
                                              duration=duration, duration_len=duration_len,
                                              feat=feat, feat_len=feat_len, pitch=pitch, pitch_len=pitch_len,
                                              energy=energy, energy_len=energy_len,
                                              spk_feat=spk_feat, spk_ids=spk_ids,
                                              rand_spk_feat=rand_spk_feat, epoch=epoch,
                                              min_frame_num=min_frame_num, max_frame_num=max_frame_num,
                                              duration_alpha=duration_alpha, energy_alpha=energy_alpha,
                                              pitch_alpha=pitch_alpha)

        # initialize the TTS output to be the decoder predictions
        outputs = dict(
            pred_feat_before=pred_feat_before, pred_feat_after=pred_feat_after, pred_feat_len=pred_feat_len,
            tgt_feat=tgt_feat, tgt_feat_len=tgt_feat_len,
            pred_pitch=pred_pitch, tgt_pitch=tgt_pitch, tgt_pitch_len=tgt_pitch_len,
            pred_energy=pred_energy, tgt_energy=tgt_energy, tgt_energy_len=tgt_energy_len,
            pred_duration=pred_duration, tgt_duration=tgt_duration, tgt_duration_len=tgt_duration_len,
        )

        def shrink_attention(input_att_list):
            # pick up the target attention layers
            if self.return_att_layer_num != -1 and len(input_att_list) > self.return_att_layer_num:
                input_att_list = input_att_list[-self.return_att_layer_num:]
            # pick up the target attention heads
            if self.return_att_head_num != -1 and input_att_list[0].size(1) > self.return_att_head_num:
                input_att_list = [att[:, :self.return_att_head_num] for att in input_att_list]
            return input_att_list

        # return the attention results if specified
        if return_att:
            # encoder self-attention
            if enc_attmat is not None and 'enc' in self.return_att_type:
                outputs.update(
                    att=dict(
                        enc=shrink_attention(enc_attmat)
                    )
                )
            # decoder self-attention
            if dec_attmat is not None and 'dec' in self.return_att_type:
                outputs['att'].update(
                    dec=shrink_attention(dec_attmat)
                )
        return outputs

    def criterion_forward(self,
                          pred_feat_before: torch.Tensor,
                          pred_feat_after: torch.Tensor,
                          tgt_feat: torch.Tensor,
                          tgt_feat_len: torch.Tensor,
                          pred_pitch: torch.Tensor,
                          tgt_pitch: torch.Tensor,
                          tgt_pitch_len: torch.Tensor,
                          pred_energy: torch.Tensor,
                          tgt_energy: torch.Tensor,
                          tgt_energy_len: torch.Tensor,
                          pred_duration: torch.Tensor,
                          tgt_duration: torch.Tensor,
                          tgt_duration_len: torch.Tensor,
                          feat_loss_fn: LeastError = None,
                          pitch_loss_fn: LeastError = None,
                          energy_loss_fn: LeastError = None,
                          duration_loss_fn: LeastError = None,
                          **kwargs) -> \
            (Dict[str, torch.Tensor], Dict[str, torch.Tensor]) or Dict[str, torch.Tensor]:

        # --- Losses Calculation --- #
        # the external feature loss function has the higher priority
        if feat_loss_fn is None:
            feat_loss_fn = self.feat_loss
        # acoustic feature prediction loss
        feat_loss_before = feat_loss_fn(pred=pred_feat_before, tgt=tgt_feat, tgt_len=tgt_feat_len)
        feat_loss_after = feat_loss_fn(pred=pred_feat_after, tgt=tgt_feat, tgt_len=tgt_feat_len)

        # the external pitch loss function has the higher priority
        if pitch_loss_fn is None:
            pitch_loss_fn = self.pitch_loss
        # pitch prediction loss
        pitch_loss = pitch_loss_fn(pred=pred_pitch, tgt=tgt_pitch, tgt_len=tgt_pitch_len)

        # the external energy loss function has the higher priority
        if energy_loss_fn is None:
            energy_loss_fn = self.energy_loss
        # energy prediction loss
        energy_loss = energy_loss_fn(pred=pred_energy, tgt=tgt_energy, tgt_len=tgt_energy_len)

        # the external duration loss function has the higher priority
        if duration_loss_fn is None:
            duration_loss_fn = self.duration_loss
        # duration prediction loss, convert the target duration into the log domain
        # note: predicted duration is already in the log domain
        duration_loss = duration_loss_fn(pred=pred_duration, tgt=torch.log(tgt_duration.float() + 1),
                                         tgt_len=tgt_duration_len)

        # combine all losses into the final one
        loss = feat_loss_before + feat_loss_after + pitch_loss + energy_loss + duration_loss
        losses = dict(loss=loss)
        # .clone() here prevents the trainable variables from value modification
        metrics = dict(loss=loss.clone().detach(),
                       feat_loss_before=feat_loss_before.clone().detach(),
                       feat_loss_after=feat_loss_after.clone().detach(),
                       pitch_loss=pitch_loss.clone().detach(),
                       energy_loss=energy_loss.clone().detach(),
                       duration_loss=duration_loss.clone().detach())

        if self.training:
            return losses, metrics
        else:
            return metrics

    def visualize(self,
                  epoch: int,
                  sample_index: str,
                  snapshot_interval: int = 1,
                  epoch_records: Dict = None,
                  domain: str = None,
                  feat: torch.Tensor = None,
                  feat_len: torch.Tensor = None,
                  pitch: torch.Tensor = None,
                  pitch_len: torch.Tensor = None,
                  text: torch.Tensor = None,
                  text_len: torch.Tensor = None,
                  duration: torch.Tensor = None,
                  duration_len: torch.Tensor = None,
                  spk_ids: torch.Tensor = None,
                  spk_feat: torch.Tensor = None):

        if len(self.visual_infer_conf) == 0:
            self.visual_infer_conf = dict(teacher_forcing=False, return_wav=False, return_feat=True)

        # obtain the inference results
        infer_results = self.inference(infer_conf=self.visual_infer_conf, return_att=True,
                                       text=text, text_len=text_len, duration=duration, duration_len=duration_len,
                                       feat=feat, feat_len=feat_len, pitch=pitch, pitch_len=pitch_len,
                                       spk_ids=spk_ids, spk_feat=spk_feat)

        # --- snapshot the objective metrics --- #
        vis_logs = []

        # --- snapshot the subjective metrics --- #
        # record the input audio and real text at the first snapshotting step
        if epoch // snapshot_interval == 1:
            # # if the audio source is raw/wav
            # if feat.size(-1) == 1:
            #     vis_logs.append(
            #         dict(
            #             plot_type='audio', materials=dict(real_wav=copy.deepcopy(feat[0])),
            #             sample_rate=self.sample_rate, audio_format=self.audio_format, subfolder_names=sample_index
            #         )
            #     )
            # # if the audio source is audio feature (mel spectrogram etc)
            # else:
            #     vis_logs.append(
            #         dict(
            #             plot_type='matrix',
            #             materials=dict(real_feat=copy.deepcopy(feat[0])),
            #             epoch=epoch, sep_save=True, sum_save=False, data_save=True, flip_y=True,
            #             subfolder_names=sample_index
            #         )
            #     )

            # snapshot input text
            vis_logs.append(
                dict(
                    materials=dict(real_text=[copy.deepcopy(self.tokenizer.tensor2text(text[0][1: -1]))]),
                    plot_type='text', subfolder_names=sample_index
                )
            )

        # snapshot the generated hypothesis acoustic features into a heatmap
        vis_logs.append(
            dict(
                plot_type='matrix', materials=dict(hypo_feat=infer_results['feat']['content'][0].transpose()),
                epoch=epoch, sep_save=False, sum_save=True, data_save=True, flip_y=True,
                subfolder_names=[sample_index, 'hypo_feat']
            )
        )

        # snapshot the predicted duration into a string
        if 'duration' not in epoch_records[sample_index].keys():
            epoch_records[sample_index]['duration'] = []
        epoch_records[sample_index]['duration'].append(str(infer_results['duration']['content'][0]))
        # snapshot the information in the materials
        vis_logs.append(
            dict(
                materials=dict(hypo_duration=copy.deepcopy(epoch_records[sample_index]['duration'])),
                plot_type='text', epoch=epoch, x_stride=snapshot_interval, subfolder_names=sample_index
            )
        )

        # hypothesis attention matrix
        infer_results['att'] = self.attention_reshape(infer_results['att'])
        self.matrix_snapshot(vis_logs=vis_logs, hypo_attention=copy.deepcopy(infer_results['att']),
                             subfolder_names=sample_index, epoch=epoch)
        return vis_logs

    @staticmethod
    def generate_ctrl_alpha(text: torch.Tensor, alpha: float, alpha_min: float, alpha_max: float,
                            ctrl_duration: bool, ctrl_energy: bool, ctrl_pitch: bool, ctrl_level: str):
        # initialize the alpha of duration for controllable TTS
        duration_alpha = None
        if ctrl_duration:
            # random alpha
            if alpha == 1.0:
                # minus 2 to remove the influence of sos and eos in text
                duration_alpha = torch.rand(
                    size=(text.size(0), 1) if ctrl_level == 'utterance' else (text.size(0), text.size(1) - 2))
                duration_alpha = (alpha_max - alpha_min) * duration_alpha + alpha_min
            # fixed alpha
            else:
                duration_alpha = torch.tensor([alpha for _ in range(text.size(0))]).unsqueeze(-1)
            # ensure the device consistency
            if text.is_cuda:
                duration_alpha = duration_alpha.cuda(text.device)

        # initialize the alpha of energy for controllable TTS
        energy_alpha = None
        if ctrl_energy:
            # random alpha
            if alpha == 1.0:
                # minus 2 to remove the influence of sos and eos in text
                energy_alpha = torch.rand(
                    size=(text.size(0), 1) if ctrl_level == 'utterance' else (text.size(0), text.size(1) - 2))
                energy_alpha = (alpha_max - alpha_min) * energy_alpha + alpha_min
            # fixed alpha
            else:
                energy_alpha = torch.tensor([alpha for _ in range(text.size(0))]).unsqueeze(-1)
            # ensure the device consistency
            if text.is_cuda:
                energy_alpha = energy_alpha.cuda(text.device)

        # initialize the alpha of pitch for controllable TTS
        pitch_alpha = None
        if ctrl_pitch:
            # random alpha
            if alpha == 1.0:
                # minus 2 to remove the influence of sos and eos in text
                pitch_alpha = torch.rand(
                    size=(text.size(0), 1) if ctrl_level == 'utterance' else (text.size(0), text.size(1) - 2))
                pitch_alpha = (alpha_max - alpha_min) * pitch_alpha + alpha_min
            # fixed alpha
            else:
                pitch_alpha = torch.tensor([alpha for _ in range(text.size(0))]).unsqueeze(-1)
            # ensure the device consistency
            if text.is_cuda:
                pitch_alpha = pitch_alpha.cuda(text.device)

        return duration_alpha, pitch_alpha, energy_alpha

    def inference(self,
                  infer_conf: Dict,
                  text: torch.Tensor = None,
                  text_len: torch.Tensor = None,
                  feat: torch.Tensor = None,
                  feat_len: torch.Tensor = None,
                  pitch: torch.Tensor = None,
                  pitch_len: torch.Tensor = None,
                  duration: torch.Tensor = None,
                  duration_len: torch.Tensor = None,
                  spk_ids: torch.Tensor = None,
                  spk_feat: torch.Tensor = None,
                  spk_feat_ids: List[str] = None,
                  domain: str = None,
                  return_att: bool = False,
                  return_feat: bool = False,
                  return_wav: bool = True,
                  use_before: bool = False,
                  teacher_forcing: bool = False) -> Dict[str, Dict[str, str or List]]:

        assert text is not None and text_len is not None

        # --- 0. Hyperparameter & Model Preparation Stage --- #
        # in-place replace infer_conf with its copy to protect the original information
        infer_conf = copy.deepcopy(infer_conf)
        # teacher_forcing in infer_conf has the higher priority and will not be passed to self.module_forward()
        if 'teacher_forcing' in infer_conf.keys():
            teacher_forcing = infer_conf.pop('teacher_forcing')
        if teacher_forcing:
            assert feat is not None and feat_len is not None and duration is not None and duration_len is not None, \
                f"If you want to decode {self.__class__.__name__} by the teacher-forcing technique, " \
                f"please give 'feat' and 'duration' in your data_cfg['test']!"

        # use_before in infer_conf has the higher priority and will not be passed to self.module_forward()
        if 'use_before' in infer_conf.keys():
            use_before = infer_conf.pop('use_before')

        # return_wav in infer_conf has the higher priority and will not be passed to self.module_forward()
        if 'return_wav' in infer_conf.keys():
            return_wav = infer_conf.pop('return_wav')
        # return_feat in infer_conf has the higher priority and will not be passed to self.module_forward()
        if 'return_feat' in infer_conf.keys():
            return_feat = infer_conf.pop('return_feat')
        assert return_wav or return_feat, "return_wav and return_feat cannot be False at the same time."

        # return_sr in infer_conf has the higher priority and will not be passed to self.module_forward()
        return_sr = None
        if 'return_sr' in infer_conf.keys():
            return_sr = infer_conf.pop('return_sr')
            assert return_sr < self.sample_rate, \
                f"You should input 'return_sr' lower than the one of the model {self.sample_rate}, " \
                f"but got return_sr={return_sr}!"
            if not hasattr(self, 'resampler'):
                self.resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=return_sr)
                if text.is_cuda:
                    self.resampler = self.resampler.cuda(text.device)

        min_frame_num = infer_conf.pop('min_frame_num') if 'min_frame_num' in infer_conf.keys() else 0
        max_frame_num = infer_conf.pop('max_frame_num') if 'max_frame_num' in infer_conf.keys() else 50

        # arguments for controllable TTS if teacher-forcing is not used
        if not teacher_forcing:
            alpha = infer_conf.pop('alpha') if 'alpha' in infer_conf.keys() else 1.0
            alpha_min = infer_conf.pop('alpha_min') if 'alpha_min' in infer_conf.keys() else 1.0
            alpha_max = infer_conf.pop('alpha_max') if 'alpha_max' in infer_conf.keys() else 1.0
            assert (alpha == 1.0) or (alpha_min == 1.0 and alpha_max == 1.0), \
                "(1) set alpha to a non-one float number; " \
                "(2) set alpha_min and/or alpha_max to non-one float numbers.\n" \
                "You can only do one of them if you want to use controllable FastSpeech2, " \
                f"but got alpha={alpha}, alpha_min={alpha_min}, alpha_max={alpha_max}!"
            assert alpha_min <= alpha_max, \
                f"alpha_min cannot be larger than alpha_max! Got alpha_min={alpha_min} and alpha_max={alpha_max}!"

            ctrl_duration = infer_conf.pop('ctrl_duration') if 'ctrl_duration' in infer_conf.keys() else False
            ctrl_energy = infer_conf.pop('ctrl_energy') if 'ctrl_energy' in infer_conf.keys() else False
            ctrl_pitch = infer_conf.pop('ctrl_pitch') if 'ctrl_pitch' in infer_conf.keys() else False
            if (alpha != 1.0) ^ (alpha_min != 1.0 or alpha_max != 1.0):
                assert ctrl_duration or ctrl_energy or ctrl_pitch, \
                    "If you want to use controllable FastSpeech2, " \
                    "please set at least one of the arguments 'ctrl_duration', 'ctrl_energy', 'ctrl_pitch' to True!"

            ctrl_level = infer_conf.pop('ctrl_level') if 'ctrl_level' in infer_conf.keys() else 'utterance'
            assert ctrl_level in ['utterance', 'token'], \
                f"The argument ctrl_level should be either 'utterance' or 'token', but got ctrl_level={ctrl_level}!"
            if ctrl_level == 'token' and alpha != 1.0:
                raise ValueError("If you want to control TTS in the level of tokens, "
                                 "please use the arguments 'alpha_min' and 'alpha_max' instead of 'alpha'.")

            duration_alpha, pitch_alpha, energy_alpha = self.generate_ctrl_alpha(
                text=text, alpha=alpha, alpha_min=alpha_min, alpha_max=alpha_max,
                ctrl_duration=ctrl_duration, ctrl_pitch=ctrl_pitch, ctrl_energy=ctrl_energy, ctrl_level=ctrl_level
            )
        # no controllable TTS is done if teacher-forcing is used
        else:
            duration_alpha, energy_alpha, pitch_alpha = None, None, None

        if len(infer_conf) != 0:
            raise RuntimeError(f"There are some unknown keys in infer_conf: {list(infer_conf.keys())}")

        # initialize the hypothesis variables
        hypo_feat, hypo_feat_len, hypo_duration, feat_token_len_ratio, hypo_att = None, None, None, None, None

        # Multi-speaker TTS scenario
        rand_spk_feat = False
        if hasattr(self.decoder, 'spk_emb'):
            batch_size = text.size(0)
            # close-set multi-speaker TTS
            if hasattr(self, 'idx2spk'):
                # randomly pick up training speakers as the reference speakers
                if spk_ids is None:
                    if not hasattr(self, 'spk_freq_dict'):
                        self.spk_freq_dict = {s_id: 0 for s_id in range(1, len(self.idx2spk) + 1)}
                    spk_ids, self.spk_freq_dict = get_min_indices_by_freq(
                        self.spk_freq_dict, chosen_idx_num=batch_size, freq_weights=to_cpu(text_len))
                    spk_ids = torch.LongTensor(spk_ids).to(text.device)

            # open-set multi-speaker TTS
            else:
                # use random vectors as the reference speaker embedding if spk_feat is not given
                if spk_feat is None:
                    # the random spk_feat obey normal distribution
                    spk_feat = torch.randn((batch_size, self.decoder.spk_emb.spk_emb_dim), device=text.device)
                    spk_feat_ids = ['rand_spk' for _ in range(batch_size)]
                    rand_spk_feat = True

        # copy the input data in advance for data safety
        model_input, outputs = copy.deepcopy(dict(text=text, text_len=text_len)), dict()
        # remove the sos at the beginning and eos at the end after copying
        text_len -= 2

        # Self Decoding or Teacher Forcing
        infer_results = self.module_forward(duration=duration if teacher_forcing else None,
                                            duration_len=duration_len if teacher_forcing else None,
                                            feat=feat if teacher_forcing else None,
                                            feat_len=feat_len if teacher_forcing else None,
                                            pitch=pitch if teacher_forcing else None,
                                            pitch_len=pitch_len if teacher_forcing else None,
                                            spk_feat=spk_feat, spk_ids=spk_ids,
                                            rand_spk_feat=rand_spk_feat, return_att=return_att,
                                            min_frame_num=min_frame_num, max_frame_num=max_frame_num,
                                            duration_alpha=duration_alpha, energy_alpha=energy_alpha,
                                            pitch_alpha=pitch_alpha, **model_input)
        # return the attention matrices
        if return_att:
            hypo_att = infer_results['att']

        # return the teacher-forcing criterion
        if teacher_forcing:
            criterion_results = self.criterion_forward(**infer_results)
            outputs.update(
                {cri_name: dict(format='txt', content=to_cpu(tensor_result))
                 for cri_name, tensor_result in criterion_results.items()}
            )
        else:
            # pred_duration is the duration in log scale
            hypo_duration = infer_results['tgt_duration']
            hypo_duration = [hypo_duration[i][:text_len[i]].type(torch.int) for i in range(len(hypo_duration))]
            # convert the duration into integers for the hard regulation
            hypo_duration = [h_d.type(torch.int) for h_d in hypo_duration]
            outputs.update(
                duration=dict(format='txt', content=to_cpu(hypo_duration))
            )

        hypo_feat = infer_results['pred_feat_before' if use_before else 'pred_feat_after']
        hypo_feat_len = infer_results['tgt_feat_len' if teacher_forcing else 'pred_feat_len']
        # hypo_feat & hypo_feat_len recovery by reduction_factor
        if self.reduction_factor > 1:
            batch_size, feat_dim = hypo_feat.size(0), hypo_feat.size(-1)
            hypo_feat = hypo_feat.reshape(
                batch_size, hypo_feat.size(1) * self.reduction_factor, feat_dim // self.reduction_factor
            )
            hypo_feat_len *= self.reduction_factor

        # denormalize the acoustic feature if needed
        if hasattr(self.decoder, 'feat_normalize'):
            hypo_feat = self.decoder.feat_normalize.recover(hypo_feat, group_ids=spk_ids)

        # turn the tensor-like spk_ids (preprocessed by self.spk2idx) into a list
        if isinstance(spk_ids, torch.Tensor):
            spk_ids = [self.idx2spk[s_id.item()] if s_id != 0 else 'aver_spk' for s_id in spk_ids]

        # calculate the Frame-to-Token ratio
        feat_token_len_ratio = hypo_feat_len / (text_len + 1e-10)

        # convert the acoustic features back to GL waveforms if specified
        if return_wav:
            try:
                hypo_wav, hypo_wav_len = self.vocode_func(hypo_feat, hypo_feat_len)
            # do not save waveforms if there is a RuntimeError
            except RuntimeError:
                pass
            # save waveforms if no error happen
            else:
                # remove the redundant silence parts at the end of the synthetic waveforms
                hypo_wav = [hypo_wav[i][:hypo_wav_len[i]] if return_sr is None else
                            self.resampler(hypo_wav[i][:hypo_wav_len[i]]) for i in range(len(hypo_wav))]
                hypo_wav_len = [wav.size(0) for wav in hypo_wav]

                # the sampling rate of the waveforms will be changed to return_sr
                outputs[f'{self.vocoder}_wav'] = dict(format='wav',
                                                      sample_rate=self.sample_rate if return_sr is None else return_sr,
                                                      group_ids=spk_ids, content=to_cpu(hypo_wav, tgt='numpy'))
                outputs[f'{self.vocoder}_wav_len'] = dict(format='txt', content=to_cpu(hypo_wav_len))

        # return the acoustic features if specified
        if return_feat:
            # remove the redundant silence parts at the end of the synthetic frames
            hypo_feat = [hypo_feat[i][:hypo_feat_len[i]] for i in range(len(hypo_feat))]
            outputs.update(
                # the sampling rate of the acoustic features remain the one of the TTS model
                feat=dict(format='npz', sample_rate=self.sample_rate,
                          group_ids=spk_ids, content=to_cpu(hypo_feat, tgt='numpy')),
                feat_len=dict(format='txt', content=to_cpu(hypo_feat_len)),
            )
        outputs.update(
            feat_token_len_ratio=dict(format='txt', content=to_cpu(feat_token_len_ratio))
        )
        # record the alpha values for controllable TTS
        if duration_alpha is not None:
            outputs.update(
                duration_alpha=dict(format='txt',
                                    content=[d_a[0] if len(d_a) == 1 else str([round(i, 2) for i in d_a])
                                             for d_a in to_cpu(duration_alpha)]))
        if energy_alpha is not None:
            outputs.update(
                energy_alpha=dict(format='txt',
                                  content=[e_a[0] if len(e_a) == 1 else str([round(i, 2) for i in e_a])
                                           for e_a in to_cpu(energy_alpha)]))
        if pitch_alpha is not None:
            outputs.update(
                pitch_alpha=dict(format='txt',
                                 content=[p_a[0] if len(p_a) == 1 else str([round(i, 2) for i in p_a])
                                          for p_a in to_cpu(pitch_alpha)]))

        # record the speaker ID used as the reference
        if spk_ids is not None:
            outputs.update(
                ref_spk=dict(format='txt', content=spk_ids)
            )
        # record the speaker embedding ID used as the reference
        if spk_feat_ids is not None:
            outputs.update(
                ref_spk_feat=dict(format='txt', content=spk_feat_ids)
            )

        # evaluation reports for all the testing instances
        instance_report_dict = {}
        # loop each utterance
        for i in range(len(text)):
            if 'Feature-Token Length Ratio' not in instance_report_dict.keys():
                instance_report_dict['Feature-Token Length Ratio'] = []
            instance_report_dict['Feature-Token Length Ratio'].append(f"{feat_token_len_ratio[i]:.2f}")

            if 'Feature Length' not in instance_report_dict.keys():
                instance_report_dict['Feature Length'] = []
            instance_report_dict['Feature Length'].append(f"{hypo_feat_len[i]:d}")
        # register the instance reports for generating instance_reports.md
        self.register_instance_reports(md_list_dict=instance_report_dict)

        # add the attention matrix into the output Dict, only used for model visualization during training
        # because it will consume too much time for saving the attention matrices of all testing samples during testing
        if return_att:
            outputs.update(
                att=hypo_att
            )
        return outputs
