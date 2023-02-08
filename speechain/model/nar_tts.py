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

from speechain.tokenizer.char import CharTokenizer
from speechain.tokenizer.g2p import GraphemeToPhonemeTokenizer
from speechain.model.abs import Model

from speechain.module.encoder.tts import TTSEncoder
from speechain.module.decoder.nar_tts import FastSpeech2Decoder

from speechain.criterion.least_error import LeastError

from speechain.utilbox.train_util import text2tensor_and_len
from speechain.utilbox.data_loading_util import parse_path_args
from speechain.utilbox.tensor_util import to_cpu


class FastSpeech2(Model):
    """
        NonAuto-Regressive FastSpeech2 Text-To-Speech Synthesis Model. (single-speaker & multi-speaker)

    """

    def module_init(self,
                    token_type: str,
                    token_vocab: str,
                    enc_emb: Dict,
                    encoder: Dict,
                    pitch_predictor: Dict,
                    energy_predictor: Dict,
                    duration_predictor: Dict,
                    decoder: Dict,
                    enc_prenet: Dict = None,
                    dec_postnet: Dict = None,
                    feat_frontend: Dict = None,
                    feat_normalize: Dict = None,
                    pitch_normalize: Dict = None,
                    energy_normalize: Dict = None,
                    spk_list: str = None,
                    spk_emb: Dict = None,
                    sample_rate: int = 22050,
                    audio_format: str = 'wav',
                    reduction_factor: int = 1,
                    len_regulation_type: str = 'floor',
                    return_att_type: List[str] or str = None,
                    return_att_head_num: int = 2,
                    return_att_layer_num: int = 2):

        # --- 1. Model-Customized Part Initialization --- #
        # initialize the tokenizer
        if token_type == 'char':
            self.tokenizer = CharTokenizer(token_vocab, copy_path=self.result_path)
        elif token_type == 'mfa':
            self.tokenizer = GraphemeToPhonemeTokenizer(token_vocab, copy_path=self.result_path)
        else:
            raise ValueError(f"Unknown token type {token_type}. "
                             f"Currently, {self.__class__.__name__} supports one of ['char', 'mfa'].")

        # initialize the speaker list if given
        if spk_list is not None:
            spk_list = np.loadtxt(parse_path_args(spk_list), dtype=str)
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
            feat_frontend['conf']['sr'] = self.sample_rate
        # update the sampling rate into the TTS Model object
        self.sample_rate = feat_frontend['conf']['sr']

        # check the speaker embedding configuration
        if spk_emb is not None:
            # multi-speaker embedding mode
            spk_emb_mode = 'open' if spk_list is None else 'close'
            if 'spk_emb_mode' in spk_emb.keys() and spk_emb['spk_emb_mode'] != spk_emb_mode:
                warnings.warn("Your input spk_emb_mode is different from the one generated by codes. "
                              "It's probably because you want to train an open-set multi-speaker TTS but "
                              "mistakenly give spk_list, or you want to train a close-set multi-speaker TTS but "
                              f"forget to give spk_list. Currently, the spk_emb_mode is set to {spk_emb_mode}.")
            # the one automatically generated by the model has the higher priority
            spk_emb['spk_emb_mode'] = spk_emb_mode

            # speaker number for the close-set multi-speaker TTS
            if spk_emb['spk_emb_mode'] == 'close':
                if 'spk_num' in spk_emb.keys() and spk_emb['spk_num'] != len(self.spk2idx) + 1:
                    warnings.warn("Your input spk_num is different from the number of speakers in your given spk_list. "
                                  f"Currently, the spk_num is set to {len(self.spk2idx) + 1}.")
                # all seen speakers plus an unknown speaker
                spk_emb['spk_num'] = len(self.spk2idx) + 1

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
            len_regulation_type=len_regulation_type,
            reduction_factor=self.reduction_factor,
        )

    @staticmethod
    def bad_cases_selection_init_fn() -> List[List[str or int]] or None:
        return [
            ['feat_token_len_ratio', 'max', 30],
            ['feat_token_len_ratio', 'min', 30],
            ['feat_len', 'max', 30],
            ['feat_len', 'min', 30]
        ]

    def batch_preprocess_fn(self, batch_data: Dict):

        def process_strings(data_dict: Dict):
            """
            turn the text strings into tensors and get their lengths

            """
            # --- Process the Text String and its Length --- #
            if 'text' in data_dict.keys():
                assert isinstance(data_dict['text'], List)
                data_dict['text'], data_dict['text_len'] = text2tensor_and_len(
                    text_list=data_dict['text'], text2tensor_func=self.tokenizer.text2tensor,
                    ignore_idx=self.tokenizer.ignore_idx
                )

            return data_dict

        # check whether the batch_data is made by multiple dataloaders
        leaf_flags = [not isinstance(value, Dict) for value in batch_data.values()]
        if sum(leaf_flags) == 0:
            return {key: process_strings(value) for key, value in batch_data.items()}
        elif sum(leaf_flags) == len(batch_data):
            return process_strings(batch_data)
        else:
            raise RuntimeError

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
                       text: torch.Tensor,
                       text_len: torch.Tensor,
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
                       epoch: int = None,
                       return_att: bool = False,
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
            kwargs:
                Temporary register used to store the redundant arguments.

        Returns:
            A dictionary containing all the TTS model outputs (feature, eos bernouli prediction) necessary to calculate the losses

        """
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
        # duration checking
        if duration is not None and duration_len is not None:
            assert duration.size(0) == text.size(0), \
                "The amounts of durations and text are not equal to each other."
            assert duration_len.size(0) == text_len.size(0), \
                "The amounts of durations and text lengths are not equal to each other."
        elif (duration is None) ^ (duration_len is None):
            raise RuntimeError(f"In {self.__class__.__name__}, "
                               f"duration and duration_len must be None or not None at the same time! "
                               f"But got duration={duration} and duration_len={duration_len}.")
        # text checking
        assert text_len.size(0) == text.size(0), \
            "The amounts of sentences and their lengths are not equal to each other."

        # remove the <sos/eos> at the beginning and the end of each sentence
        for i in range(text_len.size(0)):
            text[i, text_len[i] - 1] = self.tokenizer.ignore_idx
        text, text_len = text[:, 1:-1], text_len - 2
        # check the length of duration and text
        assert False not in [len(text[i]) == len(duration[i]) for i in range(len(text))], \
            "The lengths of individual duration and text data don't match with each other."

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
                                              spk_feat=spk_feat, spk_ids=spk_ids, rand_spk_feat=rand_spk_feat,
                                              epoch=epoch)

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

        # visualization inference is default to be done by teacher-forcing
        if len(self.visual_infer_conf) == 0:
            self.visual_infer_conf = dict(teacher_forcing=True)

        # obtain the inference results
        infer_results = self.inference(infer_conf=self.visual_infer_conf, return_att=True,
                                       text=text, text_len=text_len, duration=duration, duration_len=duration_len,
                                       feat=feat, feat_len=feat_len, pitch=pitch, pitch_len=pitch_len,
                                       spk_ids=spk_ids, spk_feat=spk_feat)

        # --- snapshot the objective metrics --- #
        vis_logs = []
        # numerical metrics recording
        materials = dict()
        for metric in ['feat_loss_before', 'feat_loss_after', 'pitch_loss', 'energy_loss', 'duration_loss']:
            # store each target metric into materials
            if metric not in epoch_records[sample_index].keys():
                epoch_records[sample_index][metric] = []
            epoch_records[sample_index][metric].append(infer_results[metric]['content'][0])
            materials[metric] = epoch_records[sample_index][metric]
        # save the visualization log
        vis_logs.append(
            dict(
                plot_type='curve', materials=copy.deepcopy(materials), epoch=epoch,
                xlabel='epoch', x_stride=snapshot_interval,
                sep_save=False, subfolder_names=sample_index
            )
        )

        # --- snapshot the subjective metrics --- #
        # record the input audio and real text at the first snapshotting step
        if epoch // snapshot_interval == 1:
            # if the audio source is raw/wav
            if feat.size(-1) == 1:
                vis_logs.append(
                    dict(
                        plot_type='audio', materials=dict(real_wav=copy.deepcopy(feat[0])),
                        sample_rate=self.sample_rate, audio_format=self.audio_format, subfolder_names=sample_index
                    )
                )
            # if the audio source is audio feature (mel spectrogram etc)
            else:
                vis_logs.append(
                    dict(
                        plot_type='matrix',
                        materials=dict(real_feat=copy.deepcopy(feat[0])),
                        epoch=epoch, sep_save=True, sum_save=False, data_save=True, flip_y=True,
                        subfolder_names=sample_index
                    )
                )

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

        # hypothesis attention matrix
        infer_results['att'] = self.attention_reshape(infer_results['att'])
        self.matrix_snapshot(vis_logs=vis_logs, hypo_attention=copy.deepcopy(infer_results['att']),
                             subfolder_names=sample_index, epoch=epoch)
        return vis_logs

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
                  use_before: bool = False,
                  teacher_forcing: bool = False) -> Dict[str, Dict[str, str or List]]:
        """

        Args:
            # --- TTS decoding arguments --- #
            infer_conf:
            # --- Testing data arguments --- #
            text:
            text_len:
            feat:
            feat_len:
            pitch:
            pitch_len:
            duration:
            duration_len:
            spk_ids:
            spk_feat:
            spk_feat_ids:
            # --- General inference arguments --- #
            domain:
            return_att:
            use_before:
            teacher_forcing:

        """
        assert text is not None and text_len is not None

        # --- 0. Hyperparameter & Model Preparation Stage --- #
        # in-place replace infer_conf with its copy to protect the original information
        infer_conf = copy.deepcopy(infer_conf)
        # teacher_forcing in infer_conf has the higher priority and will not be passed to auto_regression()
        if 'teacher_forcing' in infer_conf.keys():
            teacher_forcing = infer_conf.pop('teacher_forcing')
        if teacher_forcing:
            assert feat is not None and feat_len is not None and duration is not None and duration_len is not None, \
                f"If you want to decode {self.__class__.__name__} by the teacher-forcing technique, " \
                f"please give 'feat' and 'duration' in your data_cfg['test']!"
        else:
            assert feat is None and feat_len is None and duration is None and duration_len is None, \
                f"If you want to decode {self.__class__.__name__} without the teacher-forcing technique, " \
                f"please don't give 'feat' and 'duration' in your data_cfg['test']!"

        # use_before in infer_conf has the higher priority than the default values
        if 'use_before' in infer_conf.keys():
            use_before = infer_conf['use_before']

        hypo_feat, hypo_feat_len, feat_token_len_ratio, hypo_att = None, None, None, None

        # Multi-speaker TTS scenario
        rand_spk_feat = False
        if hasattr(self.decoder, 'spk_emb'):
            batch_size = text.size(0)
            # close-set multi-speaker TTS
            if hasattr(self, 'idx2spk'):
                # randomly pick up training speakers as the reference speakers
                if spk_ids is None:
                    spk_ids = torch.randint(low=1, high=len(self.idx2spk) + 1, size=(batch_size,), device=text.device)
            # open-set multi-speaker TTS
            else:
                # use random vectors as the reference speaker embedding if spk_feat is not given
                if spk_feat is None:
                    # make sure that the range of random speaker feature is [-1, 1)
                    spk_feat = torch.rand((batch_size, self.decoder.spk_emb.spk_emb_dim), device=text.device) * 2 - 1
                    spk_feat_ids = ['rand_spk' for _ in range(batch_size)]
                    rand_spk_feat = True

        # copy the input data in advance for data safety
        model_input, outputs = copy.deepcopy(dict(text=text, text_len=text_len)), dict()

        # LM Decoding by Teacher Forcing
        infer_results = self.module_forward(duration=duration if teacher_forcing else None,
                                            duration_len=duration_len if teacher_forcing else None,
                                            feat=feat if teacher_forcing else None,
                                            feat_len=feat_len if teacher_forcing else None,
                                            pitch=pitch if teacher_forcing else None,
                                            pitch_len=pitch_len if teacher_forcing else None,
                                            spk_feat=spk_feat, spk_ids=spk_ids,
                                            rand_spk_feat=rand_spk_feat, return_att=return_att, **model_input)
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

        hypo_feat = infer_results['pred_feat_before' if use_before else 'pred_feat_after']
        hypo_feat_len = infer_results['tgt_feat_len' if teacher_forcing else 'pred_feat_len']
        # hypo_feat & hypo_feat_len recovery by reduction_factor
        if self.reduction_factor > 1:
            batch_size, feat_dim = hypo_feat.size(0), hypo_feat.size(-1)
            hypo_feat = hypo_feat.reshape(
                batch_size, hypo_feat.size(1) * self.reduction_factor, feat_dim // self.reduction_factor
            )
            hypo_feat_len *= self.reduction_factor

        # denormalize the acoustic feature if needed #
        if hasattr(self.decoder, 'feat_normalize'):
            hypo_feat = self.decoder.feat_normalize.recover(hypo_feat, group_ids=spk_ids)

        # remove the sos at the beginning and eos at the end
        feat_token_len_ratio = hypo_feat_len / (text_len - 2)

        # remove the redundant silence parts at the end of the synthetic frames
        hypo_feat = [hypo_feat[i][:hypo_feat_len[i]] for i in range(len(hypo_feat))]
        outputs.update(
            # the sampling rate of the generated waveforms is obtained from the frontend of the decoder
            feat=dict(format='npz', sample_rate=self.sample_rate, content=to_cpu(hypo_feat, tgt='numpy')),
            feat_len=dict(format='txt', content=to_cpu(hypo_feat_len)),
            feat_token_len_ratio=dict(format='txt', content=to_cpu(feat_token_len_ratio))
        )

        # record the speaker ID used as the reference
        if spk_ids is not None:
            assert hasattr(self, 'idx2spk')
            outputs.update(
                ref_spk=dict(format='txt',
                             content=[self.idx2spk[s_id.item()] if s_id != 0 else 0 for s_id in spk_ids])
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
