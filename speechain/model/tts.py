"""
    Author: Sashi Novitasari
    Affiliation: NAIST
    Date: 2022.08

    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.09
"""
import copy
import math
import warnings

import numpy as np
import torch

from typing import Dict, Any, List

from speechain.model.abs import Model
from speechain.tokenizer.char import CharTokenizer
from speechain.utilbox.train_util import make_mask_from_len, text2tensor_and_len, spk2tensor
from speechain.utilbox.data_loading_util import parse_path_args

from speechain.module.encoder.tts import TTSEncoder
from speechain.module.decoder.tts import TTSDecoder

from speechain.criterion.least_error import LeastError
from speechain.criterion.bce_logits import BCELogits
from speechain.criterion.accuracy import Accuracy
from speechain.criterion.fbeta_score import FBetaScore

from speechain.infer_func.tts_decoding import auto_regression
from speechain.utilbox.tensor_util import to_cpu


class TTS(Model):
    """
    Encoder-decoder-based single/multi speaker TTS
    """

    def module_init(self,
                    token_type: str,
                    token_vocab: str,
                    frontend: Dict,
                    enc_emb: Dict,
                    enc_prenet: Dict,
                    encoder: Dict,
                    dec_prenet: Dict,
                    decoder: Dict,
                    dec_postnet: Dict,
                    normalize: Dict = None,
                    vocoder: Dict = None,
                    spk_list: str = None,
                    spk_emb: Dict = None,
                    sample_rate: int = 22050,
                    audio_format: str = 'wav',
                    reduction_factor: int = 1,
                    stop_threshold: float = 0.5):
        """
        Args:
            # --- module_conf arguments --- #
            frontend: Dict (mandatory)
                The configuration of the acoustic feature extraction frontend in the `TTSDecoder` member.
                This argument must be given since our toolkit doesn't support time-domain TTS.
                For more details about how to give `frontend`, please refer to speechain.module.encoder.tts.TTSDecoder.
            normalize: Dict
                The configuration of the normalization layer in the `TTSDecoder` member.
                This argument can also be given as a bool value.
                True means the default configuration and False means no normalization.
                For more details about how to give `normalize`, please refer to
                    speechain.module.norm.feat_norm.FeatureNormalization.
            enc_emb: Dict (mandatory)
                The configuration of the embedding layer in the `TTSEncoder` member.
                The encoder prenet embeds the input token id into token embeddings before feeding them into
                the encoder.
                For more details about how to give `enc_emb`, please refer to speechain.module.encoder.tts.TTSEncoder.
            enc_prenet: Dict (mandatory)
                The configuration of the prenet in the `TTSEncoder` member.
                The encoder prenet embeds the input token embeddings into high-level embeddings before feeding them into
                the encoder.
                For more details about how to give `enc_prent`, please refer to speechain.module.encoder.tts.TTSEncoder.
            encoder: Dict (mandatory)
                The configuration of the encoder main body in the `TTSEncoder` member.
                The encoder embeds the input embeddings into the encoder representations at each time steps of the
                input acoustic features.
                For more details about how to give `encoder`, please refer to speechain.module.encoder.tts.TTSEncoder.
            spk_emb: Dict = None (conditionally mandatory)
                The configuration for the `SPKEmbedPrenet` in the `TTSDecoder` member.
                For more details about how to give `spk_emb`, please refer to
                    speechain.module.prenet.spk_embed.SpeakerEmbedPrenet.
            dec_prenet: Dict (mandatory)
                The configuration of the prenet in the `TTSDecoder` member.
                For more details about how to give `dec_prenet`, please refer to speechain.module.encoder.tts.TTSDecoder.
            decoder: Dict (mandatory)
                The configuration of the decoder main body in the `TTSDecoder` member.
                For more details about how to give `decoder`, please refer to speechain.module.decoder.tts.TTSDecoder.
            dec_postnet: Dict (mandatory)
                The configuration of the postnet in the `TTSDecoder` member.
                For more details about how to give `dec_postnet`, please refer to speechain.module.encoder.tts.TTSDecoder.
            vocoder: Dict (optional)
                The configuration of the vocoder member.
            # --- customize_conf arguments --- #
            token_type: (mandatory)
                The type of the built-in tokenizer.
                Currently, we support 'char' for `CharTokenizer` and 'phn' for `PhonemeTokenizer`.
            token_vocab: (mandatory)
                The path of the vocabulary list `vocab` for initializing the built-in tokenizer.
            spk_list: str = None (conditionally mandatory)
                The path of the speaker list that contains all the speaker ids in your training set.
                If you would like to train a close-set multi-speaker TTS, you need to give a spk_list.
            sample_rate: int = 22050 (optional)
                The sampling rate of the target speech.
                Currently it's used for acoustic feature extraction frontend initialization and tensorboard register of
                the input speech during model visualization.
                In the future, this argument will also be used to dynamically downsample the input speech during training.
            audio_format: str = 'wav' (optional)
                This argument is only used for input speech recording during model visualization.
            reduction_factor: int = 1 (mandatory)
                The factor that controls how much the length of output speech feature is reduced.
            stop_threshold: float = 0.5 (mandatory)
                The threshold that controls whether the speech synthesis stops or not.
        """
        # --- 1. Model-Customized Part Initialization --- #
        # initialize the tokenizer
        if token_type == 'char':
            self.tokenizer = CharTokenizer(token_vocab)
        elif token_type == 'phn':
            raise NotImplementedError
        else:
            raise ValueError

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
        self.stop_threshold = stop_threshold

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
        if 'sr' not in frontend['conf'].keys():
            frontend['conf']['sr'] = self.sample_rate
        else:
            if frontend['conf']['sr'] != self.sample_rate:
                warnings.warn(
                    "The sampling rate in your frontend configuration doesn't match the one in customize_conf. "
                    "The one in your configuration will be used to extract acoustic features.")

        # check the speaker embedding configuration
        if spk_emb is not None:
            # multi-speaker embedding mode
            spk_emb_mode = 'open' if spk_list is None else 'close'
            if 'spk_emb_mode' in spk_emb.keys() and spk_emb['spk_emb_mode'] != spk_emb_mode:
                warnings.warn("Your input spk_emb_mode is different from the one generated by codes. "
                              "It's probably because you want to train an open-set multi-speaker TTS but "
                              "mistakenly give spk_list, or you want to train a close-set multi-speaker TTS but "
                              "forget to give spk_list. "
                              f"Currently, the spk_emb_mode is set to {spk_emb_mode}.")
            # the one automatically generated by the model has the higher priority
            spk_emb['spk_emb_mode'] = spk_emb_mode

            # speaker number for the close-set multi-speaker TTS
            if spk_emb['spk_emb_mode'] == 'close':
                if 'spk_num' in spk_emb.keys() and spk_emb['spk_num'] != len(self.spk2idx) + 1:
                    warnings.warn("Your input spk_num is different from the number of speakers in your given spk_list. "
                                  f"Currently, the spk_num is set to {len(self.spk2idx) + 1}.")
                # all seen speakers plus an unknown speaker
                spk_emb['spk_num'] = len(self.spk2idx) + 1

        self.decoder = TTSDecoder(
            spk_emb=spk_emb,
            frontend=frontend,
            normalize=normalize,
            prenet=dec_prenet,
            decoder=decoder,
            postnet=dec_postnet,
            distributed=self.distributed,
            reduction_factor=self.reduction_factor,
        )

        # --- 2.3. Vocoder construction --- #
        if vocoder is None:
            self.vocoder = self.decoder.spec_to_wav
        else:
            raise NotImplementedError("Neural vocoders are not supported yet/(ToT)/~~")

    @staticmethod
    def bad_cases_selection_init_fn() -> List[List[str or int]] or None:
        return [
            ['feat_token_len_ratio', 'max', 30],
            ['feat_token_len_ratio', 'min', 30],
            ['wav_len_ratio', 'max', 30],
            ['wav_len_ratio', 'min', 30]
        ]

    def criterion_init(self,
                       feat_loss_type: str = 'L2',
                       feat_loss_norm: bool = True,
                       feat_update_range: int or float = None,
                       stop_pos_weight: float = 5.0,
                       stop_loss_norm: bool = True,
                       f_beta: int = 2):
        """
        This function initializes all the necessary _Criterion_ members:
            1. `speechain.criterion.least_error.LeastError` for acoustic feature prediction loss calculation.
            2. `speechain.criterion.bce_logits.BCELogits` for stop flag prediction loss calculation.
            3. `speechain.criterion.accuracy.Accuracy` for teacher-forcing stop flag prediction accuracy calculation.
            4. `speechain.criterion.fbeta_score.FBetaScore` for teacher-forcing stop flag prediction f-score calculation.

        Args:
            feat_loss_type: str = 'L2'
                The type of acoustic feature prediction loss. Should be either 'L1', 'L2', and 'L1+L2'.
                For more details, please refer to speechain.criterion.least_error.LeastError.
            feat_loss_norm: bool = True
                Controls whether the sentence normalization is performed for feature loss calculation.
                For more details, please refer to speechain.criterion.least_error.LeastError.
            feat_update_range: int or float = None
                The updating range of the dimension of acoustic features for feature loss calculation.
                For more details, please refer to speechain.criterion.least_error.LeastError.
            stop_pos_weight: float = 5.0
                The weight putted on stop points for stop loss calculation.
                For more details, please refer to speechain.criterion.bce_logits.BCELogits.
            stop_loss_norm: bool = True
                Controls whether the sentence normalization is performed for stop loss calculation.
                For more details, please refer to speechain.criterion.bce_logits.BCELogits.
            f_beta: int = 2
                The value of beta for stop flag f-score calculation.
                The larger beta is, the more f-score focuses on true positive stop flag prediction result.
                For more details, please refer to speechain.criterion.fbeta_score.FBetaScore.

        """
        # --- Criterion Part Initialization --- #
        # training loss
        self.feat_loss = LeastError(loss_type=feat_loss_type, is_normalized=feat_loss_norm,
                                    update_range=feat_update_range)
        self.stop_loss = BCELogits(pos_weight=stop_pos_weight, is_normalized=stop_loss_norm)
        # validation metrics
        self.stop_accuracy = Accuracy()
        self.stop_fbeta = FBetaScore(beta=f_beta)

    def batch_preprocess_fn(self, batch_data: Dict):
        """

        Args:
            batch_data:

        Returns:

        """

        def process_strings(data_dict: Dict):
            """
            turn the text and speaker strings into tensors and get their lengths

            Args:
                data_dict:

            Returns:

            """
            # --- Process the Text String and its Length --- #
            if 'text' in data_dict.keys():
                assert isinstance(data_dict['text'], List)
                data_dict['text'], data_dict['text_len'] = text2tensor_and_len(
                    text_list=data_dict['text'], text2tensor_func=self.tokenizer.text2tensor,
                    ignore_idx=self.tokenizer.ignore_idx
                )

            # --- Process the Speaker ID String --- #
            if 'spk_ids' in data_dict.keys() and hasattr(self, 'spk2idx'):
                assert isinstance(data_dict['spk_ids'], List)
                data_dict['spk_ids'] = spk2tensor(spk_list=data_dict['spk_ids'], spk2idx_dict=self.spk2idx)

            return data_dict

        # check whether the batch_data is made by multiple dataloaders
        leaf_flags = [not isinstance(value, Dict) for value in batch_data.values()]
        if sum(leaf_flags) == 0:
            return {key: process_strings(value) for key, value in batch_data.items()}
        elif sum(leaf_flags) == len(batch_data):
            return process_strings(batch_data)
        else:
            raise RuntimeError

    def module_forward(self,
                       feat: torch.Tensor,
                       text: torch.Tensor,
                       feat_len: torch.Tensor,
                       text_len: torch.Tensor,
                       spk_feat: torch.Tensor = None,
                       spk_ids: torch.Tensor = None,
                       epoch: int = None,
                       return_att: bool = None,
                       return_hidden: bool = None,
                       return_enc: bool = None,
                       **kwargs) -> Dict[str, torch.Tensor]:
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
            spk_feat: (batch, 1, speaker embedding dim)
                Pre-extracted speaker embedding. (None means single-speaker TTS)
            spk_ids: (batch,)
                The speaker ids of each speech data. In the form of integer values.
            epoch: int
                The number of the current training epoch.
                Mainly used for mean&std calculation in the feature normalization
            return_att: bool
                Controls whether the attention matrices of each layer in the encoder and decoder will be returned.
            return_hidden: bool
                Controls whether the hidden representations of each layer in the encoder and decoder will be returned.
            return_enc: bool
                Controls whether the final encoder representations will be returned.
            kwargs:
                Temporary register used to store the redundant arguments.

        Returns:
            A dictionary containing all the TTS model outputs (feature, eos bernouli prediction) necessary to calculate the losses

        """
        # para checking
        assert feat.size(0) == text.size(0) and feat_len.size(0) == text_len.size(0), \
            "The amounts of utterances and sentences are not equal to each other."
        assert feat_len.size(0) == feat.size(0), \
            "The amounts of utterances and their lengths are not equal to each other."
        assert text_len.size(0) == text.size(0), \
            "The amounts of sentences and their lengths are not equal to each other."

        # Encoding, we don't remove the <sos/eos> at the beginning and end of the sentence
        enc_text, enc_text_mask, enc_attmat, enc_hidden = self.encoder(text=text, text_len=text_len)

        # Decoding
        pred_stop, pred_feat_before, pred_feat_after, tgt_feat, tgt_feat_len, dec_attmat, encdec_attmat, dec_hidden \
            = self.decoder(enc_text=enc_text, enc_text_mask=enc_text_mask, feat=feat, feat_len=feat_len,
                           spk_feat=spk_feat, spk_ids=spk_ids, epoch=epoch)

        # initialize the TTS output to be the decoder predictions
        outputs = dict(
            pred_stop=pred_stop,
            pred_feat_before=pred_feat_before,
            pred_feat_after=pred_feat_after,
            tgt_feat=tgt_feat,
            tgt_feat_len=tgt_feat_len
        )

        # return the attention results of either encoder or decoder if specified
        if return_att:
            outputs.update(
                att=dict(
                    enc_att=enc_attmat,
                    dec_att=dict(
                        self_att=dec_attmat,
                        encdec_att=encdec_attmat
                    )
                )
            )
        # return the internal hidden results of both encoder and decoder if specified
        if return_hidden:
            outputs.update(
                hidden=dict(
                    enc_hidden=enc_hidden,
                    dec_hidden=dec_hidden
                )
            )
        # return the encoder outputs if specified
        if return_enc:
            outputs.update(
                enc_feat=enc_text,
                enc_feat_mask=enc_text_mask
            )
        return outputs

    def criterion_forward(self,
                          pred_stop: torch.Tensor,
                          pred_feat_before: torch.Tensor,
                          pred_feat_after: torch.Tensor,
                          tgt_feat: torch.Tensor,
                          tgt_feat_len: torch.Tensor,
                          **kwargs) -> \
            (Dict[str, torch.Tensor], Dict[str, torch.Tensor]) or Dict[str, torch.Tensor]:
        """

        Args:
            pred_stop: (batch, seq_len, 1)
                predicted stop probability
            pred_feat_before: (batch, seq_len, feat_dim * reduction_factor)
                predicted acoustic feature before postnet residual addition
            pred_feat_after: (batch, seq_len, feat_dim * reduction_factor)
                predicted acoustic feature after postnet residual addition
            tgt_feat: (batch, seq_len, feat_dim)
                processed acoustic features, length-reduced and edge-padded
            tgt_feat_len: (batch,)
            **kwargs:

        Returns:
            losses:
            metric:

        """
        # --- Losses Calculation --- #
        # acoustic feature prediction loss
        feat_loss_before = self.feat_loss(pred=pred_feat_before, tgt=tgt_feat, tgt_len=tgt_feat_len)
        feat_loss_after = self.feat_loss(pred=pred_feat_after, tgt=tgt_feat, tgt_len=tgt_feat_len)

        # feature prediction stop loss
        pred_stop = pred_stop.squeeze(-1)
        tgt_stop = (1. - make_mask_from_len(
            tgt_feat_len - 1, max_len=tgt_feat_len.max().item(), mask_type=torch.float
        ).squeeze(1))
        if pred_stop.is_cuda:
            tgt_stop = tgt_stop.cuda(pred_stop.device)
        # end-flag prediction
        stop_loss = self.stop_loss(pred=pred_stop, tgt=tgt_stop, tgt_len=tgt_feat_len)

        # combine all losses into the final one
        loss = feat_loss_before + feat_loss_after + stop_loss

        # --- Metrics Calculation --- #
        logits_threshold = -math.log(1 / self.stop_threshold - 1)
        pred_stop_hard = pred_stop > logits_threshold
        stop_accuracy = self.stop_accuracy(pred_stop_hard, tgt_stop, tgt_feat_len)
        stop_fbeta = self.stop_fbeta(pred_stop_hard, tgt_stop, tgt_feat_len)

        losses = dict(loss=loss)
        # .clone() here prevents the trainable variables from value modification
        metrics = dict(loss=loss.clone().detach(),
                       feat_loss_before=feat_loss_before.clone().detach(),
                       feat_loss_after=feat_loss_after.clone().detach(),
                       stop_loss=stop_loss.clone().detach(),
                       stop_accuracy=stop_accuracy.detach())
        metrics[f"stop_f{self.stop_fbeta.beta}"] = stop_fbeta.detach()

        if self.training:
            return losses, metrics
        else:
            return metrics

    def matrix_snapshot(self, vis_logs: List, hypo_attention: Dict, subfolder_names: List[str] or str, epoch: int):
        """
        recursively snapshot all the attention matrices

        Args:
            hypo_attention:
            subfolder_names:

        Returns:

        """
        if isinstance(subfolder_names, str):
            subfolder_names = [subfolder_names]
        keys = list(hypo_attention.keys())

        # process the input data by different data types
        if isinstance(hypo_attention[keys[0]], Dict):
            for key, value in hypo_attention.items():
                self.matrix_snapshot(vis_logs=vis_logs, hypo_attention=value,
                                     subfolder_names=subfolder_names + [key], epoch=epoch)

        # snapshot the information in the materials
        elif isinstance(hypo_attention[keys[0]], np.ndarray):
            vis_logs.append(
                dict(
                    plot_type='matrix', materials=hypo_attention, epoch=epoch,
                    sep_save=False, data_save=False, subfolder_names=subfolder_names
                )
            )

    def attention_reshape(self, hypo_attention: Dict, prefix_list: List = None) -> Dict:
        """

        Args:
            hypo_attention:
            prefix_list:

        """
        if prefix_list is None:
            prefix_list = []

        # process the input data by different data types
        if isinstance(hypo_attention, Dict):
            return {key: self.attention_reshape(value, prefix_list + [key]) for key, value in hypo_attention.items()}
        elif isinstance(hypo_attention, List):
            return {str(index): self.attention_reshape(element, prefix_list + [str(index)])
                    for index, element in enumerate(hypo_attention)}
        elif isinstance(hypo_attention, torch.Tensor):
            hypo_attention = hypo_attention.squeeze()
            if hypo_attention.is_cuda:
                hypo_attention = hypo_attention.detach().cpu()

            if hypo_attention.dim() == 2:
                return {'.'.join(prefix_list + [str(0)]): hypo_attention.numpy()}
            elif hypo_attention.dim() == 3:
                return {'.'.join(prefix_list + [str(index)]): element.numpy()
                        for index, element in enumerate(hypo_attention)}
            else:
                raise RuntimeError

    def visualize(self,
                  epoch: int,
                  sample_index: str,
                  snapshot_interval: int,
                  epoch_records: Dict,
                  feat: torch.Tensor,
                  feat_len: torch.Tensor,
                  text: torch.Tensor,
                  text_len: torch.Tensor,
                  speaker_feat: torch.Tensor = None):
        """

        Args:
            epoch:
            feat:
            feat_len:
            text:
            text_len:
            speaker_feat: (optional)

        Returns:

        """
        # obtain the inference results
        infer_results = self.inference(feat=feat, feat_len=feat_len,
                                       text=text, text_len=text_len,
                                       speaker_feat=speaker_feat,
                                       return_att=True, **self.visual_infer_conf)

        # --- snapshot the objective metrics --- #
        vis_logs = []
        # CER, WER, hypothesis probability
        materials = dict()
        for metric in ['loss_total', 'feat_loss', 'bern_loss', 'bern_accuracy', 'bern_f1']:
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
            # snapshot target audio
            if self.speechfeat_generator is not None:  # if the audio source is raw/wav
                vis_logs.append(
                    dict(
                        plot_type='audio', materials=dict(target_audio=copy.deepcopy(feat[0].cpu().numpy())),
                        sample_rate=self.sample_rate, audio_format=self.audio_format, subfolder_names=sample_index
                    )
                )
            else:  # if the audio source is audio feature (mel spectrogram etc)
                seqlen, featdim = feat[0].size()
                feat_tgt_ungroup = feat[0].reshape(seqlen * self.speechfeat_group, featdim // self.speechfeat_group)
                vis_logs.append(
                    dict(
                        plot_type='matrix',
                        materials=dict(target_feat=copy.deepcopy(feat_tgt_ungroup.transpose(0, 1).cpu().numpy())),
                        epoch=epoch,
                        sep_save=False, data_save=False, subfolder_names=sample_index
                    )
                )

            # snapshot input text
            vis_logs.append(
                dict(
                    materials=dict(real_text=[copy.deepcopy(self.tokenizer.tensor2text(text[0][1: -1]))]),
                    plot_type='text', subfolder_names=sample_index
                )
            )

        # hypothesis feature
        if 'hypo_feat_npz' not in epoch_records[sample_index].keys():
            epoch_records[sample_index]['hypo_feat_npz'] = []
        epoch_records[sample_index]['hypo_feat_npz'].append(infer_results['hypo_feat_npz']['content'][0])
        # snapshot the information in the materials

        if self.speechfeat_generator is not None:  # if the generated speech is raw/wav
            vis_logs.append(
                dict(
                    materials=dict(hypo_feat=copy.deepcopy(infer_results['hypo_feat_npz']['content'][0])),
                    plot_type='audio', epoch=epoch, x_stride=snapshot_interval,
                    subfolder_names=sample_index
                ))

        else:  # if the generated speech is speech feature
            vis_logs.append(
                dict(
                    plot_type='matrix',
                    materials=dict(hypo_feat=copy.deepcopy(np.transpose(infer_results['hypo_feat_npz']['content'][0]))),
                    epoch=epoch,
                    sep_save=False, data_save=False, subfolder_names=sample_index
                )
            )

        # hypothesis attention matrix
        infer_results['hypo_att'] = self.attention_reshape(infer_results['hypo_att'])
        self.matrix_snapshot(vis_logs=vis_logs, hypo_attention=copy.deepcopy(infer_results['hypo_att']),
                             subfolder_names=sample_index, epoch=epoch)
        return vis_logs

    def inference(self,
                  infer_conf: Dict,
                  text: torch.Tensor = None,
                  text_len: torch.Tensor = None,
                  feat: torch.Tensor = None,
                  feat_len: torch.Tensor = None,
                  spk_ids: torch.Tensor = None,
                  spk_feat: torch.Tensor = None,
                  aver_spk: bool = False,
                  return_att: bool = False,
                  feat_only: bool = False,
                  decode_only: bool = False,
                  use_dropout: bool = False,
                  use_before: bool = False,
                  teacher_forcing: bool = False) -> Dict[str, Any]:
        """

        Args:
            # --- Testing data arguments --- #
            feat: (batch_size, feat_maxlen, feat_dim)
                The ground-truth utterance for the input text
                Used for teacher-forcing decoding and objective evaluation
            feat_len: (batch_size,)
                The length of `feat`.
            text: (batch_size, text_maxlen)
                The text data to be inferred.
            text_len: (batch_size,)
                The length of `text`.
            spk_ids: (batch_size,)
                The ID of the reference speaker.
            spk_feat: (batch_size,)
                The speaker embedding of the reference speaker.
            # --- General inference arguments --- #
            aver_spk: bool = False
                Whether you use the average speaker as the reference speaker.
                The speaker embedding of the average speaker is a zero vector.
            return_att: bool = False
                Whether the attention matrix of the input speech is returned.
            feat_only: bool = False
                Whether only decode the text into acoustic features without vocoding.
            decode_only: bool = False
                Whether skip the evaluation step and do the decoding step only.
            use_dropout: bool = False
                Whether turn on the dropout layers in the prenet of the TTS decoder when decoding.
            teacher_forcing: bool = False
                Whether turn on the dropout layers in the prenet of the TTS decoder when decoding.
            # --- TTS decoding arguments --- #
            infer_conf:
                The inference configuration given from the `infer_cfg` in your `exp_cfg`.
                For more details, please refer to speechain.infer_func.tts_decoding.auto_regression.

        """
        assert text is not None and text_len is not None

        # --- 0. Hyperparameter & Model Preparation Stage --- #
        # in-place replace infer_conf with its copy to protect the original information
        infer_conf = copy.deepcopy(infer_conf)
        # The following argumentsin infer_conf has the higher priority and will not be passed to auto_regression()
        if 'decode_only' in infer_conf.keys():
            decode_only = infer_conf.pop('decode_only')
        if 'teacher_forcing' in infer_conf.keys():
            teacher_forcing = infer_conf.pop('teacher_forcing')
        if 'use_dropout' in infer_conf.keys():
            use_dropout = infer_conf.pop('use_dropout')
        if 'aver_spk' in infer_conf.keys():
            aver_spk = infer_conf.pop('aver_spk')
        if 'feat_only' in infer_conf.keys():
            feat_only = infer_conf.pop('feat_only')

        # 'stop_threshold', and 'use_before' are kept as the arguments of auto_regression()
        # stop_threshold in infer_conf has the higher priority than the built-in one of the model
        if 'stop_threshold' not in infer_conf.keys():
            infer_conf['stop_threshold'] = self.stop_threshold
        # feat_only in infer_conf has the higher priority than the default values
        if 'use_before' in infer_conf.keys():
            use_before = infer_conf['use_before']
        else:
            infer_conf['use_before'] = use_before

        hypo_feat, hypo_feat_len, hypo_len_ratio, hypo_wav, hypo_wav_len, hypo_wav_len_ratio, hypo_att = \
            None, None, None, None, None, None, None

        # turn the dropout layer in the decoder on for introducing variability to the synthetic utterances
        if use_dropout:
            self.decoder.turn_on_dropout()

        # set the speaker embedding to zero vectors for multi-speaker TTS
        if aver_spk and hasattr(self.decoder, 'spk_emb'):
            spk_feat = torch.zeros((text.size(0), self.decoder.spk_emb.spk_emb_dim), device=text.device)
            spk_ids = None

        # --- 1. Acoustic Feature Generation Stage --- #
        outputs = dict()
        # --- 1.1. The 1st Pass: TTS Auto-Regressive Decoding --- #
        if not teacher_forcing:
            # copy the input data in advance for data safety
            model_input = copy.deepcopy(dict(text=text, text_len=text_len))

            # Encoding input text
            enc_text, enc_text_mask, _, _ = self.encoder(**model_input)

            # Generate the synthetic acoustic features auto-regressively
            infer_results = auto_regression(enc_text=enc_text,
                                            enc_text_mask=enc_text_mask,
                                            spk_ids=spk_ids,
                                            spk_feat=spk_feat,
                                            reduction_factor=self.reduction_factor,
                                            feat_dim=self.decoder.output_size,
                                            decode_one_step=self.decoder,
                                            **infer_conf)
            hypo_feat = infer_results['hypo_feat']
            hypo_feat_len = infer_results['hypo_feat_len']
            hypo_len_ratio = infer_results['hypo_len_ratio']
            outputs.update(feat_token_len_ratio=dict(format='txt', content=to_cpu(hypo_len_ratio)))

        # --- 1.2. The 2nd Pass: TTS Teacher-Forcing Decoding --- #
        if teacher_forcing or return_att:
            infer_results = self.module_forward(feat=feat if teacher_forcing else hypo_feat,
                                                feat_len=feat_len if teacher_forcing else hypo_feat_len,
                                                text=text, text_len=text_len,
                                                spk_feat=spk_feat, spk_ids=spk_ids,
                                                return_att=return_att)
            # return the attention matrices
            if return_att:
                hypo_att = infer_results['att']

            # update the hypothesis feature-related data in the teacher forcing mode
            if teacher_forcing:
                hypo_feat = infer_results['pred_feat_before' if use_before else 'pred_feat_after']
                hypo_feat_len = feat_len
                hypo_len_ratio = hypo_feat_len / text_len
                outputs.update(self.criterion_forward(**infer_results))

        # --- 1.3. The 3rd Pass: denormalize the acoustic feature if needed --- #
        if hasattr(self.decoder, 'normalize'):
            hypo_feat = self.decoder.normalize.recover(hypo_feat, group_ids=spk_ids)

        # --- 2. Post-processing for the Generated Acoustic Features --- #
        # recover the waveform from the acoustic feature by the vocoder
        if not feat_only:
            assert self.vocoder is not None, \
                "Please specify a vocoder if you want to recover the waveform from the acoustic features."
            hypo_wav, hypo_wav_len = self.vocoder(feat=hypo_feat, feat_len=hypo_feat_len)
            hypo_wav = [hypo_wav[i][:hypo_wav_len[i]] for i in range(len(hypo_wav))]
            outputs = dict(
                # the sampling rate of the generated waveforms is obtained from the frontend of the decoder
                wav=dict(format=self.audio_format, sample_rate=self.decoder.frontend.get_sample_rate(),
                         content=to_cpu(hypo_wav, tgt='numpy')),
                wav_len=dict(format='txt', content=to_cpu(hypo_wav_len))
            )

        # remove the redundant silence parts at the end of the synthetic frames
        hypo_feat = [hypo_feat[i][:hypo_feat_len[i]] for i in range(len(hypo_feat))]
        outputs.update(
            feat=dict(format='npy', content=to_cpu(hypo_feat, tgt='numpy')),
            feat_len=dict(format='txt', content=to_cpu(hypo_feat_len))
        )

        # record the speaker ID used as the reference
        if spk_ids is not None:
            assert hasattr(self, 'idx2spk')
            outputs.update(
                ref_spk=dict(format='txt',
                             content=[self.idx2spk[s_id.item()] if s_id != 0 else 0 for s_id in spk_ids])
            )

        # add the attention matrix into the output Dict, only used for model visualization during training
        # because it will consume too much time for saving the attention matrices of all testing samples during testing
        if return_att:
            outputs.update(
                hypo_att=hypo_att
            )

        # --- 3. Supervised Metrics Calculation (Ground-Truth is involved here) --- #
        # hypo_wav_len_ratio is a supervised metrics calculated by the decoded waveforms
        if not decode_only:
            # For the acoustic feature ground-truth, since the transformation from feature length to waveform length
            # is linear, acoustic length ratio is equal to waveform length ratio.
            hypo_wav_len_ratio = hypo_feat_len / feat_len if feat.size(-1) != 1 else hypo_wav_len / feat_len
            outputs.update(wav_len_ratio=dict(format='txt', content=to_cpu(hypo_wav_len_ratio)))

        # --- 4. Final Report Generation Stage --- #
        # produce the sample report by a .md file
        instance_report_dict = dict()
        for i in range(len(hypo_feat)):
            # add the feat2token len_ratio into instance_reports.md
            if 'Hypothesis Confidence' not in instance_report_dict.keys():
                instance_report_dict['Feature-Token Length Ratio'] = []
            instance_report_dict['Feature-Token Length Ratio'].append(f"{hypo_len_ratio[i]:.2f}")

            # udpate the supervised metrics into the current sample report
            if not decode_only:
                if 'Hypo-Real Waveform Length Ratio' not in instance_report_dict.keys():
                    instance_report_dict['Hypo-Real Waveform Length Ratio'] = []
                instance_report_dict['Hypo-Real Waveform Length Ratio'].append(
                    f"{hypo_wav_len_ratio[i]:.2%} "
                    f"({'-' if hypo_wav_len_ratio[i] < 1 else '+'}{abs(hypo_wav_len_ratio[i] - 1):.2%})")

        # register the instance reports for generating instance_reports.md
        self.register_instance_reports(md_list_dict=instance_report_dict)
        return outputs

