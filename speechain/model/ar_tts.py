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
from speechain.tokenizer.g2p import GraphemeToPhonemeTokenizer
from speechain.utilbox.train_util import make_mask_from_len, text2tensor_and_len, spk2tensor
from speechain.utilbox.data_loading_util import parse_path_args

from speechain.module.encoder.tts import TTSEncoder
from speechain.module.decoder.ar_tts import ARTTSDecoder

from speechain.criterion.least_error import LeastError
from speechain.criterion.bce_logits import BCELogits
from speechain.criterion.att_guid import AttentionGuidance
from speechain.criterion.accuracy import Accuracy
from speechain.criterion.fbeta_score import FBetaScore

from speechain.infer_func.tts_decoding import auto_regression
from speechain.utilbox.tensor_util import to_cpu


class ARTTS(Model):
    """
        Auto-Regressive Attention-based Text-To-Speech Synthesis Model. (single-speaker or multi-speaker)

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
                    spk_list: str = None,
                    spk_emb: Dict = None,
                    sample_rate: int = 22050,
                    audio_format: str = 'wav',
                    reduction_factor: int = 1,
                    stop_threshold: float = 0.5,
                    return_att_type: List[str] or str = None,
                    return_att_head_num: int = 2,
                    return_att_layer_num: int = 2):
        """

        Args:
            # --- module_conf arguments --- #
            frontend: Dict (mandatory)
                The configuration of the acoustic feature extraction frontend in the `ARTTSDecoder` member.
                This argument must be given since our toolkit doesn't support time-domain TTS.
                For more details about how to give `frontend`, please refer to speechain.module.encoder.ar_tts.ARTTSDecoder.
            normalize: Dict
                The configuration of the normalization layer in the `ARTTSDecoder` member.
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
                The configuration for the `SPKEmbedPrenet` in the `ARTTSDecoder` member.
                For more details about how to give `spk_emb`, please refer to
                    speechain.module.prenet.spk_embed.SpeakerEmbedPrenet.
            dec_prenet: Dict (mandatory)
                The configuration of the prenet in the `ARTTSDecoder` member.
                For more details about how to give `dec_prenet`, please refer to speechain.module.encoder.ar_tts.ARTTSDecoder.
            decoder: Dict (mandatory)
                The configuration of the decoder main body in the `ARTTSDecoder` member.
                For more details about how to give `decoder`, please refer to speechain.module.decoder.ar_tts.ARTTSDecoder.
            dec_postnet: Dict (mandatory)
                The configuration of the postnet in the `ARTTSDecoder` member.
                For more details about how to give `dec_postnet`, please refer to speechain.module.encoder.ar_tts.ARTTSDecoder.
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
            return_att_type: List[str] or str = 'encdec'
                The type of attentions you want to return for both attention guidance and attention visualization.
                It can be given as a string (one type) or a list of strings (multiple types).
                The type should be one of
                    1. 'encdec': the encoder-decoder attention, shared by both Transformer and RNN
                    2. 'enc': the encoder self-attention, only for Transformer
                    3. 'dec': the decoder self-attention, only for Transformer
            return_att_head_num: int = -1
                The number of returned attention heads. If -1, all the heads in an attention layer will be returned.
                RNN can be considered to one-head attention, so return_att_head_num > 1 is equivalent to 1 for RNN.
            return_att_layer_num: int = 1
                The number of returned attention layers. If -1, all the attention layers will be returned.
                RNN can be considered to one-layer attention, so return_att_layer_num > 1 is equivalent to 1 for RNN.

        """
        # --- 1. Model-Customized Part Initialization --- #
        # initialize the tokenizer
        if token_type == 'char':
            self.tokenizer = CharTokenizer(token_vocab, copy_path=self.result_path)
        elif token_type == 'g2p':
            self.tokenizer = GraphemeToPhonemeTokenizer(token_vocab, copy_path=self.result_path)
        else:
            raise ValueError(f"Unknown token type {token_type}. "
                             f"Currently, {self.__class__.__name__} supports one of ['char', 'g2p'].")

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

        if return_att_type is None:
            self.return_att_type = ['encdec', 'enc', 'dec']
        else:
            self.return_att_type = return_att_type if isinstance(return_att_type, List) else [return_att_type]
        for i in range(len(self.return_att_type)):
            if self.return_att_type[i].lower() in ['enc', 'dec', 'encdec']:
                self.return_att_type[i] = self.return_att_type[i].lower()
            else:
                raise ValueError("The elements of your input return_att_type must be one of ['enc', 'dec', 'encdec'], "
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
        if 'sr' not in frontend['conf'].keys():
            frontend['conf']['sr'] = self.sample_rate
        # update the sampling rate into the TTS Model object
        self.sample_rate = frontend['conf']['sr']

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

        self.decoder = ARTTSDecoder(
            spk_emb=spk_emb,
            frontend=frontend,
            normalize=normalize,
            prenet=dec_prenet,
            decoder=decoder,
            postnet=dec_postnet,
            distributed=self.distributed,
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

    def criterion_init(self,
                       feat_loss: Dict[str, Any] = None,
                       stop_loss: Dict[str, Any] = None,
                       att_guid_loss: Dict[str, Any] or bool = None):
        """
        This function initializes all the necessary Criterion members for an autoregressive TTS:
            1. `speechain.criterion.least_error.LeastError` for acoustic feature prediction loss calculation.
            2. `speechain.criterion.bce_logits.BCELogits` for stop flag prediction loss calculation.
            3. `speechain.criterion.accuracy.Accuracy` for teacher-forcing stop flag prediction accuracy calculation.
            4. `speechain.criterion.fbeta_score.FBetaScore` for teacher-forcing stop flag prediction f-score calculation.

        Args:
            feat_loss: Dict[str, Any]
                The arguments for LeastError(). If not given, the default setting of LeastError() will be used.
                Please refer to speechain.criterion.least_error.LeastError for more details.
            stop_loss: Dict[str, Any]
                The arguments for BCELogits(). If not given, the default setting of BCELogits() will be used.
                Please refer to speechain.criterion.bce_logits.BCELogits for more details.
            att_guid_loss: Dict[str, Any] or bool
                The arguments for AttentionGuidance(). If not given, self.att_guid_loss won't be initialized.
                This argument can also be set to a bool value 'True'. If True, the default setting of AttentionGuidance()
                will be used.
                Please refer to speechain.criterion.att_guid.AttentionGuidance for more details.

        """
        # --- Criterion Part Initialization --- #
        # training losses
        if feat_loss is None:
            feat_loss = {}
        self.feat_loss = LeastError(**feat_loss)

        if stop_loss is None:
            stop_loss = {}
        self.stop_loss = BCELogits(**stop_loss)

        if att_guid_loss is not None:
            # if att_guid_loss is given as True, the default arguments of AttentionGuidance will be used
            if not isinstance(att_guid_loss, Dict):
                assert isinstance(att_guid_loss, bool) and att_guid_loss, \
                    "If you want to use the default setting of AttentionGuidance, please give att_guid_loss as True."
                att_guid_loss = {}

            assert 'encdec' in self.return_att_type, \
                "If you want to enable attention guidance for ASR training, please include 'encdec' in return_att_type."
            self.att_guid_loss = AttentionGuidance(**att_guid_loss)

        # validation metrics
        self.stop_accuracy = Accuracy()
        self.stop_fbeta = FBetaScore(beta=2)

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
            if 'spk_ids' in data_dict.keys():
                assert isinstance(data_dict['spk_ids'], List) and hasattr(self, 'spk2idx')
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
                       return_att: bool = False,
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
        if spk_ids is None and spk_feat is not None:
            assert (not hasattr(self, 'idx2spk')) and (not hasattr(self, 'spk2idx')), \
                "If you are using the external speaker embedding (spk_feat), " \
                "please don't given spk_list in customize_conf!"

        # Encoding, we don't remove the <sos/eos> at the beginning and end of the sentence
        enc_returns = self.encoder(text=text, text_len=text_len)
        # Transformer-based encoder additionally returns the encoder self-attention
        if len(enc_returns) == 4:
            enc_text, enc_text_mask, enc_attmat, enc_hidden = enc_returns
        # RNN-based encoder doesn't return any attention
        elif len(enc_returns) == 3:
            (enc_text, enc_text_mask, enc_hidden), enc_attmat = enc_returns, None
        else:
            raise RuntimeError

        # Decoding
        dec_returns = self.decoder(enc_text=enc_text, enc_text_mask=enc_text_mask, feat=feat, feat_len=feat_len,
                                   spk_feat=spk_feat, spk_ids=spk_ids, epoch=epoch)
        # Transformer-based decoder additionally returns the decoder self-attention
        if len(dec_returns) == 8:
            pred_stop, pred_feat_before, pred_feat_after, \
            tgt_feat, tgt_feat_len, dec_attmat, encdec_attmat, dec_hidden = dec_returns
        # RNN-based decoder only returns the encoder-decoder attention
        elif len(dec_returns) == 7:
            (pred_stop, pred_feat_before, pred_feat_after,
             tgt_feat, tgt_feat_len, encdec_attmat, dec_hidden), dec_attmat = dec_returns, None
        else:
            raise RuntimeError

        # initialize the TTS output to be the decoder predictions
        outputs = dict(
            pred_stop=pred_stop,
            pred_feat_before=pred_feat_before,
            pred_feat_after=pred_feat_after,
            tgt_feat=tgt_feat,
            tgt_feat_len=tgt_feat_len
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
        if return_att or hasattr(self, 'att_guid_loss'):
            # encoder-decoder attention
            if 'encdec' in self.return_att_type:
                # register the encoder-decoder attention
                outputs.update(
                    att=dict(
                        encdec=shrink_attention(encdec_attmat)
                    )
                )
            # encoder self-attention
            if enc_attmat is not None and 'enc' in self.return_att_type:
                outputs['att'].update(
                    enc=shrink_attention(enc_attmat)
                )
            # decoder self-attention
            if dec_attmat is not None and 'dec' in self.return_att_type:
                outputs['att'].update(
                    dec=shrink_attention(dec_attmat)
                )
        return outputs

    def criterion_forward(self,
                          pred_stop: torch.Tensor,
                          pred_feat_before: torch.Tensor,
                          pred_feat_after: torch.Tensor,
                          tgt_feat: torch.Tensor,
                          tgt_feat_len: torch.Tensor,
                          text_len: torch.Tensor,
                          att: Dict[str, List[torch.Tensor]] = None,
                          feat_loss_fn: LeastError = None,
                          stop_loss_fn: BCELogits = None,
                          att_guid_loss_fn: AttentionGuidance = None,
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
            text_len: (batch,)
            att:
            feat_loss_fn:
            stop_loss_fn:
            att_guid_loss_fn:
            **kwargs:
                Unnecessary arguments for criterion calculation.

        Returns:
            losses:
            metric:

        """
        # --- Losses Calculation --- #
        # the external feature loss function has the higher priority
        if feat_loss_fn is None:
            feat_loss_fn = self.feat_loss
        # acoustic feature prediction loss
        feat_loss_before = feat_loss_fn(pred=pred_feat_before, tgt=tgt_feat, tgt_len=tgt_feat_len)
        feat_loss_after = feat_loss_fn(pred=pred_feat_after, tgt=tgt_feat, tgt_len=tgt_feat_len)

        # feature prediction stop loss
        pred_stop = pred_stop.squeeze(-1)
        tgt_stop = (1. - make_mask_from_len(
            tgt_feat_len - 1, max_len=tgt_feat_len.max().item(), mask_type=torch.float
        ).squeeze(1))
        if pred_stop.is_cuda:
            tgt_stop = tgt_stop.cuda(pred_stop.device)
        # the external feature loss function has the higher priority
        if stop_loss_fn is None:
            stop_loss_fn = self.stop_loss
        # end-flag prediction
        stop_loss = stop_loss_fn(pred=pred_stop, tgt=tgt_stop, tgt_len=tgt_feat_len)

        # combine all losses into the final one
        loss = feat_loss_before + feat_loss_after + stop_loss

        # attention guidance loss
        if att_guid_loss_fn is not None or hasattr(self, 'att_guid_loss'):
            # the external attention guidance loss function has the higher priority
            if att_guid_loss_fn is None:
                att_guid_loss_fn = self.att_guid_loss

            # layer_num * (batch, head_num, ...) -> (batch, layer_num * head_num, ...)
            att_tensor = torch.cat(att['encdec'], dim=1)
            att_guid_loss = att_guid_loss_fn(att_tensor, tgt_feat_len, text_len)
            loss += att_guid_loss
        else:
            att_guid_loss = None

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
        metrics[f"stop_f{int(self.stop_fbeta.beta)}"] = stop_fbeta.detach()
        if att_guid_loss is not None:
            metrics["att_guid_loss"] = att_guid_loss.clone().detach()

        if self.training:
            return losses, metrics
        else:
            return metrics

    def matrix_snapshot(self, vis_logs: List, hypo_attention: Dict, subfolder_names: List[str] or str, epoch: int):

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
                    sep_save=False, data_save=True, subfolder_names=subfolder_names
                )
            )

    def attention_reshape(self, hypo_attention: Dict, prefix_list: List = None) -> Dict:

        if prefix_list is None:
            prefix_list = []

        # process the input data by different data types
        if isinstance(hypo_attention, Dict):
            return {key: self.attention_reshape(value, prefix_list + [key]) for key, value in hypo_attention.items()}
        elif isinstance(hypo_attention, List):
            return {str(index - len(hypo_attention)): self.attention_reshape(
                hypo_attention[index], prefix_list + [str(index - len(hypo_attention))])
                for index in range(len(hypo_attention) - 1, -1, -1)}
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
                  snapshot_interval: int = 1,
                  epoch_records: Dict = None,
                  domain: str = None,
                  feat: torch.Tensor = None,
                  feat_len: torch.Tensor = None,
                  text: torch.Tensor = None,
                  text_len: torch.Tensor = None,
                  spk_ids: torch.Tensor = None,
                  spk_feat: torch.Tensor = None):

        # visualization inference is default to be done by teacher-forcing
        if len(self.visual_infer_conf) == 0:
            self.visual_infer_conf = dict(teacher_forcing=True)

        # obtain the inference results
        infer_results = self.inference(infer_conf=self.visual_infer_conf, return_att=True,
                                       feat=feat, feat_len=feat_len, text=text, text_len=text_len,
                                       spk_ids=spk_ids, spk_feat=spk_feat)

        # --- snapshot the objective metrics --- #
        vis_logs = []
        # numerical metrics recording
        materials = dict()
        for metric in ['loss', 'stop_accuracy', 'stop_f2']:
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
                  spk_ids: torch.Tensor = None,
                  spk_feat: torch.Tensor = None,
                  spk_feat_ids: List[str] = None,
                  domain: str = None,
                  return_att: bool = False,
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
            spk_feat: (batch_size, spk_feat_dim)
                The speaker embedding of the reference speaker.
            spk_feat_ids: List[str] = None
                The IDs for the input spk_feat. Mainly used to record the reference speaker embedding during inference.
            # --- General inference arguments --- #
            domain: str = None
                This argument indicates which domain the input speech belongs to.
                It's used to indicate the `TTSDecoder` member how to encode the input speech.
            return_att: bool = False
                Whether the attention matrix of the input speech is returned.
            use_dropout: bool = False
                Whether turn on the dropout layers in the prenet of the TTS decoder when decoding.
            use_before: bool = False
                Whether return the acoustic feature not processed by the postnet.
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
        if 'teacher_forcing' in infer_conf.keys():
            teacher_forcing = infer_conf.pop('teacher_forcing')
        if 'use_dropout' in infer_conf.keys():
            use_dropout = infer_conf.pop('use_dropout')

        # 'stop_threshold', and 'use_before' are kept as the arguments of auto_regression()
        # stop_threshold in infer_conf has the higher priority than the built-in one of the model
        if 'stop_threshold' not in infer_conf.keys():
            infer_conf['stop_threshold'] = self.stop_threshold
        # use_before in infer_conf has the higher priority than the default values
        if 'use_before' in infer_conf.keys():
            use_before = infer_conf['use_before']
        else:
            infer_conf['use_before'] = use_before

        hypo_feat, hypo_feat_len, feat_token_len_ratio, hypo_att = None, None, None, None

        # turn the dropout layer in the decoder on for introducing variability to the synthetic utterances
        if use_dropout:
            self.decoder.turn_on_dropout()

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
                                            rand_spk_feat=rand_spk_feat,
                                            reduction_factor=self.reduction_factor,
                                            feat_dim=self.decoder.output_size,
                                            decode_one_step=self.decoder,
                                            **infer_conf)
            hypo_feat = infer_results['hypo_feat']
            hypo_feat_len = infer_results['hypo_feat_len']
            feat_token_len_ratio = infer_results['feat_token_len_ratio']

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
                criterion_results = self.criterion_forward(text_len=text_len, **infer_results)
                outputs.update(
                    {cri_name: dict(format='txt', content=to_cpu(tensor_result))
                     for cri_name, tensor_result in criterion_results.items()}
                )
                hypo_feat = infer_results['pred_feat_before' if use_before else 'pred_feat_after']
                hypo_feat_len = infer_results['tgt_feat_len']
                # hypo_feat_len recovery by reduction_factor
                if self.reduction_factor > 1:
                    batch_size, feat_dim = hypo_feat.size(0), hypo_feat.size(-1)
                    hypo_feat = hypo_feat.reshape(
                        batch_size, hypo_feat.size(1) * self.reduction_factor, feat_dim // self.reduction_factor
                    )
                    hypo_feat_len *= self.reduction_factor
                feat_token_len_ratio = hypo_feat_len / text_len

        # --- 1.3. The 3rd Pass: denormalize the acoustic feature if needed --- #
        if hasattr(self.decoder, 'normalize'):
            hypo_feat = self.decoder.normalize.recover(hypo_feat, group_ids=spk_ids)

        # --- 2. Post-processing for the Generated Acoustic Features --- #
        # remove the redundant silence parts at the end of the synthetic frames
        hypo_feat = [hypo_feat[i][:hypo_feat_len[i]] for i in range(len(hypo_feat))]
        outputs.update(
            # the sampling rate of the generated waveforms is obtained from the frontend of the decoder
            feat=dict(format='npz', sample_rate=self.sample_rate, content=to_cpu(hypo_feat, tgt='numpy')),
            feat_len=dict(format='txt', content=to_cpu(hypo_feat_len)),
            # text_len=dict(format='txt', content=to_cpu(text_len)),
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


class MultiDomainARTTS(ARTTS):
    """
    Auto-Regressive TTS model trained by multiple dataloaders on different domains.

    """
    def criterion_init(self,
                       loss_weights: Dict[str, float] = None,
                       feat_loss: Dict = None,
                       stop_loss: Dict = None,
                       att_guid_loss: Dict = None,
                       **kwargs):
        """

        Args:
            loss_weights:
            feat_loss:
            stop_loss:
            att_guid_loss:
            **kwargs:

        Returns:

        """
        # register the weight for each loss if loss_weights is given
        if loss_weights is not None:
            self.loss_weights = dict()
            for loss_name, weight in loss_weights.items():
                assert isinstance(weight, float) and 0 < weight < 1, \
                    f"Your input weight should be a float number in (0, 1), but got loss_weights[{loss_name}]={weight}!"
                self.loss_weights[loss_name] = weight

        def recur_init_loss_by_dict(loss_dict: Dict, loss_class):
            leaf_num = sum([not isinstance(value, Dict) for value in loss_dict.values()])
            # all the items in loss_dict are not Dict mean that the loss function is shared by all the dataloaders
            if leaf_num == len(loss_dict):
                return loss_class(**loss_dict)
            # no item in loss_dict is Dict mean that each dataloader has its own loss function
            elif leaf_num == 0:
                if hasattr(self, 'loss_weights'):
                    assert len(loss_dict) == len(self.loss_weights), \
                        "The key number in the xxx_loss should match the one in the loss_weights"

                nested_loss = dict()
                for name, conf in loss_dict.items():
                    if hasattr(self, 'loss_weights'):
                        assert name in self.loss_weights.keys(), \
                            f"The key name {name} doesn't match anyone in the loss_weights!"
                    nested_loss[name] = loss_class(**conf)
                return nested_loss
            else:
                raise RuntimeError("Your loss configuration must be either Dict[str, Any] or Dict[str, Dict[str, Any]]")

        # feature loss will be initialized no matter whether feat_loss is given or not
        self.feat_loss = recur_init_loss_by_dict(feat_loss, LeastError) if feat_loss is not None else LeastError()

        # stop loss will be initialized no matter whether stop_loss is given or not
        self.stop_loss = recur_init_loss_by_dict(stop_loss, BCELogits) if stop_loss is not None else BCELogits()

        # only initialize attention-guidance loss if it is given
        if att_guid_loss is not None:
            assert 'encdec' in self.return_att_type, \
                "If you want to enable attention guidance for ASR training, please include 'encdec' in return_att_type."

            # if att_guid_loss is given as True, the default arguments of AttentionGuidance will be used
            if not isinstance(att_guid_loss, Dict):
                assert isinstance(att_guid_loss, bool) and att_guid_loss, \
                    "If you want to use the default setting of AttentionGuidance, please give att_guid_loss as True."

            if isinstance(att_guid_loss, Dict):
                self.att_guid_loss = recur_init_loss_by_dict(att_guid_loss, AttentionGuidance)
            # att_guid_loss is True, intialize the default AttentionGuidance criterion
            else:
                self.att_guid_loss = AttentionGuidance()

        # validation metrics
        self.stop_accuracy = Accuracy()
        self.stop_fbeta = FBetaScore(beta=2)

    def module_forward(self, **batch_data) -> Dict[str, Dict or torch.Tensor]:
        """

        Args:
            **batch_data:

        Returns:

        """
        # whether the input batch_data is generated by multiple dataloaders
        multi_flag = sum([not isinstance(value, torch.Tensor) for value in batch_data.values()]) == len(batch_data)

        # Single-dataloader scenario
        # probably for the validation stage of in-domain semi-supervised ASR where we only have one data-label pair
        if not multi_flag:
            return super(MultiDomainARTTS, self).module_forward(**batch_data)
        # Multi-dataloader scenario
        # For semi-supervised training or validation of out-domain semi-supervised ASR where we may have multiple
        # data-label pairs in a single batch, we need to go through forward function once for each pair.
        else:
            # pop the non-Dict arguments from the input batch data
            general_args, data_keys = dict(), list(batch_data.keys())
            for key in data_keys:
                if not isinstance(batch_data[key], Dict):
                    general_args[key] = batch_data.pop(key)

            # otherwise, go through the normal training process once for all the sub-batches
            # (each sub-batch corresponds to a dataloader)
            return {domain: super(MultiDomainARTTS, self).module_forward(domain=domain, **general_args, **domain_data)
                    for domain, domain_data in batch_data.items()}

    def criterion_forward(self, **data_output_dict) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
        """

        Args:
            **data_output_dict:

        Returns:

        """
        # whether the input data_output_dict is generated by multiple dataloaders
        multi_flag = sum([isinstance(value, Dict) for value in data_output_dict.values()]) == len(data_output_dict)

        # Single-dataloader scenario
        # probably for the validation stage of in-domain semi-supervised ASR where we only have one data-label pair
        if not multi_flag:
            return super(MultiDomainARTTS, self).criterion_forward(**data_output_dict)
        # Multi-dataloader scenario
        # For semi-supervised training or validation of out-domain semi-supervised ASR where we may have multiple
        # data-label pairs in a single batch, we need to go through forward function once for each pair.
        else:
            losses, metrics, domain_list = dict(), dict(), list(data_output_dict.keys())
            for domain in domain_list:
                # initialize the feature loss function
                feat_loss_fn = self.feat_loss[domain] if isinstance(self.feat_loss, Dict) else self.feat_loss
                # initialize the stop loss function
                stop_loss_fn = self.stop_loss[domain] if isinstance(self.stop_loss, Dict) else self.stop_loss
                # initialize the attention-guidance loss function only if att_guid_loss is created
                if hasattr(self, 'att_guid_loss'):
                    att_guid_loss_fn = self.att_guid_loss[domain] if isinstance(self.att_guid_loss, Dict) \
                        else self.att_guid_loss
                else:
                    att_guid_loss_fn = None

                # call the criterion_forward() of the parent class by the initialized loss functions
                _criteria = super(MultiDomainARTTS, self).criterion_forward(
                    feat_loss_fn=feat_loss_fn, stop_loss_fn=stop_loss_fn, att_guid_loss_fn=att_guid_loss_fn,
                    **data_output_dict[domain])

                # update loss and metric Dicts during training
                if self.training:
                    # update the losses and metrics Dicts by the domain name at the beginning
                    losses.update(**{f"{domain}_{_key}": _value for _key, _value in _criteria[0].items()})
                    metrics.update(**{f"{domain}_{_key}": _value for _key, _value in _criteria[1].items()})
                # only update metric Dict during validation
                else:
                    metrics.update(**{_key if len(domain_list) == 1 else f"{domain}_{_key}": _value
                                      for _key, _value in _criteria.items()})

            # calculate the overall weighted loss during training
            if self.training:
                # normalize losses of all the domains by the given loss_weights
                if hasattr(self, 'loss_weights'):
                    assert len(self.loss_weights) == len(domain_list), \
                        "There is a number mismatch of the domains between your data_cfg and train_cfg."
                    assert sum([domain in self.loss_weights.keys() for domain in domain_list]) == len(domain_list), \
                        "There is a name mismatch of the domains between your data_cfg and train_cfg."
                    losses.update(
                        loss=sum(
                            [losses[f"{domain}_loss"] * self.loss_weights[domain] for domain in domain_list]
                        ) / sum(
                            [self.loss_weights[domain] for domain in domain_list]
                        )
                    )
                # average losses of all the domains if loss_weights is not given
                else:
                    losses.update(
                        loss=sum([losses[f"{domain}_loss"] for domain in domain_list]) / len(domain_list)
                    )
                metrics.update(loss=losses['loss'].clone().detach())

            if self.training:
                return losses, metrics
            else:
                return metrics

    def inference(self, infer_conf: Dict, **test_batch) -> Dict[str, Any]:
        """

        Args:
            infer_conf:
            **test_batch:

        Returns:

        """
        multi_flag = sum([isinstance(value, Dict) for value in test_batch.values()]) == len(test_batch)
        # no sub-Dict means one normal supervised dataloader, go through the inference function of ASR
        if not multi_flag:
            return super(MultiDomainARTTS, self).inference(infer_conf=infer_conf, **test_batch)

        # sub-Dict means that the domain information is given for ASR inference
        else:
            assert len(test_batch) == 1, \
                "If you want to evaluate the ASR model by multiple domains, please evaluate them one by one."
            for domain, domain_batch in test_batch.items():
                return super(MultiDomainARTTS, self).inference(infer_conf=infer_conf, domain=domain, **domain_batch)

