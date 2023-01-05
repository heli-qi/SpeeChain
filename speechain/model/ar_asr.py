"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import copy
import warnings

import numpy as np
import torch
from typing import Dict, Any, List

from speechain.model.abs import Model
from speechain.tokenizer.char import CharTokenizer
from speechain.tokenizer.sp import SentencePieceTokenizer
from speechain.infer_func.beam_search import beam_searching
from speechain.utilbox.tensor_util import to_cpu
from speechain.utilbox.eval_util import get_word_edit_alignment
from speechain.utilbox.train_util import text2tensor_and_len

from speechain.module.encoder.asr import ASREncoder
from speechain.module.decoder.asr import ASRDecoder
from speechain.module.postnet.token import TokenPostnet

from speechain.criterion.cross_entropy import CrossEntropy
from speechain.criterion.ctc import CTCLoss
from speechain.criterion.att_guid import AttentionGuidance
from speechain.criterion.accuracy import Accuracy
from speechain.criterion.error_rate import ErrorRate


class ARASR(Model):
    """
    Auto-Regressive Attention-based Automatic Speech Recognition (AR-ASR) implementation.

    The neural network structure of an `ASR` Model object is made up of 3 Module members:

    1. an `ASREncoder` made up of:
        1. `frontend` converts the  raw waveforms into acoustic features on-the-fly.
        2. `normalize` normalizes the extracted acoustic features to normal distribution for faster convergence.
        3. `specaug` randomly warps and masks the normalized acoustic features.
        4. `prenet` preprocesses the augmented acoustic features before passing them to the encoder.
        5. `encoder` extracts the encoder hidden representations of the preprocessed acoustic features and passes them
            to `ASRDecoder`.

    2. an `ASRDecoder` made up of:
        1. `embedding` embeds each tokens in the input sentence into token embedding vectors.
        2. `decoder` extracts the decoder hidden representations based on the token embedding vectors and encoder
            hidden representations.
        3. `postnet` predicts the probability of the next tokens by the decoder hidden representations.

    3. (optional) a CTC layer made up of a 'TokenPostnet'

    """

    def module_init(self,
                    token_type: str,
                    token_vocab: str,
                    frontend: Dict,
                    enc_prenet: Dict,
                    encoder: Dict,
                    dec_emb: Dict,
                    decoder: Dict,
                    normalize: Dict or bool = None,
                    specaug: Dict or bool = None,
                    sample_rate: int = 16000,
                    audio_format: str = 'wav',
                    return_att_type: List[str] or str = None,
                    return_att_head_num: int = 2,
                    return_att_layer_num: int = 2):
        """
        This initialization function contains 4 steps:
        1. `Tokenizer` initialization.
        2. `ASREncoder` initialization.
        3. `ASRDecoder` initialization.
        4. (optional) 'CTC' layer initialization

        The input arguments of this function are two-fold:
        1. the ones from `customize_conf` of `model` in `train_cfg`
        2. the ones from `module_conf` of `model` in `train_cfg`

        Args:
            # --- module_conf arguments --- #
            frontend: (mandatory)
                The configuration of the acoustic feature extraction frontend in the `ASREncoder` member.
                This argument must be given since our toolkit doesn't support time-domain ASR.
                For more details about how to give `frontend`, please refer to speechain.module.encoder.asr.ASREncoder.
            normalize: (optional)
                The configuration of the normalization layer in the `ASREncoder` member.
                This argument can also be given as a bool value.
                True means the default configuration and False means no normalization.
                For more details about how to give `normalize`, please refer to
                    speechain.module.norm.feat_norm.FeatureNormalization.
            specaug: (optional)
                The configuration of the SpecAugment layer in the `ASREncoder` member.
                This argument can also be given as a bool value.
                True means the default configuration and False means no SpecAugment.
                For more details about how to give `specaug`, please refer to
                    speechain.module.augment.specaug.SpecAugment.
            enc_prenet: (mandatory)
                The configuration of the prenet in the `ASREncoder` member.
                The encoder prenet embeds the input acoustic features into hidden embeddings before feeding them into
                the encoder.
                For more details about how to give `enc_prent`, please refer to speechain.module.encoder.asr.ASREncoder.
            encoder: (mandatory)
                The configuration of the encoder main body in the `ASREncoder` member.
                The encoder embeds the hidden embeddings into the encoder representations at each time steps of the
                input acoustic features.
                For more details about how to give `encoder`, please refer to speechain.module.encoder.asr.ASREncoder.
            dec_emb: (mandatory)
                The configuration of the embedding layer in the `ASRDecoder` member.
                The decoder prenet embeds the input token ids into hidden embeddings before feeding them into
                the decoder.
                For more details about how to give `dec_emb`, please refer to speechain.module.encoder.asr.ASREncoder.
            decoder: (mandatory)
                The configuration of the decoder main body in the `ASRDecoder` member.
                The decoder predicts the probability of the next token at each time steps based on the token embeddings.
                For more details about how to give `decoder`, please refer to speechain.module.decoder.asr.ASRDecoder.
            # --- customize_conf arguments --- #
            token_type: (mandatory)
                The type of the built-in tokenizer.
            token_vocab: (mandatory)
                The path of the vocabulary for initializing the built-in tokenizer.
            sample_rate: int = 16000 (optional)
                The sampling rate of the input speech.
                Currently, it's used for acoustic feature extraction frontend initialization and tensorboard register of
                the input speech for model visualization.
                In the future, this argument will also be used to on-the-fly downsample the input speech.
            audio_format: (optional)
                This argument is only used for input speech recording during model visualization.
            return_att_type: List[str] or str = ['encdec', 'enc', 'dec']
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

        # --- 1. Module-independent Initialization --- #
        # initialize the tokenizer
        if token_type.lower() == 'char':
            self.tokenizer = CharTokenizer(token_vocab)
        elif token_type.lower() == 'sentencepiece':
            self.tokenizer = SentencePieceTokenizer(token_vocab)
        else:
            raise ValueError(f"Unknown token_type {token_type}. "
                             f"Your input token_type should be one of ['char', 'sentencepiece'].")

        # initialize the sampling rate, mainly used for visualizing the input audio during training
        self.sample_rate = sample_rate
        self.audio_format = audio_format.lower()

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

        # --- 2. Module Initialization --- #
        # --- 2.1. Encoder construction --- #
        # the sampling rate will be first initialized
        if 'sr' not in frontend['conf'].keys():
            frontend['conf']['sr'] = self.sample_rate
        # update the sampling rate into the ASR Model object
        self.sample_rate = frontend['conf']['sr']
        self.encoder = ASREncoder(
            frontend=frontend,
            normalize=normalize,
            specaug=specaug,
            prenet=enc_prenet,
            encoder=encoder,
            distributed=self.distributed
        )

        # --- 2.2. Decoder construction --- #
        # the vocabulary size is given by the built-in tokenizer instead of the input configuration
        if 'vocab_size' in dec_emb['conf'].keys():
            if dec_emb['conf']['vocab_size'] != self.tokenizer.vocab_size:
                warnings.warn(f"Your input vocabulary size is different from the one obtained from the built-in "
                              f"tokenizer ({self.tokenizer.vocab_size}). The latter one will be used to initialize the "
                              f"decoder for correctness.")
            dec_emb['conf'].pop('vocab_size')
        self.decoder = ASRDecoder(
            vocab_size=self.tokenizer.vocab_size,
            embedding=dec_emb,
            decoder=decoder
        )

    def criterion_init(self,
                       ce_normalized: bool = False, label_smoothing: float = 0.0,
                       ctc_weight: float = 0.0, ctc_zero_infinity: bool = True,
                       att_guid_sigma: float = 0.0):
        """
        This function initializes all the necessary _Criterion_ members:
            1. `speechain.criterion.cross_entropy.CrossEntropy` for training loss calculation.
            2. `speechain.criterion.ctc.CTCLoss` for training loss calculation.
            3. `speechain.criterion.accuracy.Accuracy` for teacher-forcing validation accuracy calculation.
            4. `speechain.criterion.error_rate.ErrorRate` for evaluation CER & WER calculation.

        Args:
            ce_normalized: bool = False
                Controls whether the sentence normalization is performed for cross-entropy loss.
                For more details, please refer to speechain.criterion.cross_entropy.CrossEntropy
            label_smoothing: float = 0.0
                Controls the scale of label smoothing. 0 means no smoothing.
                For more details, please refer to speechain.criterion.cross_entropy.CrossEntropy
            ctc_weight: float = 0.0
                The weight on the CTC loss for training the ASR model. If ctc_weight == 0, the CTC layer won't be
                created and the ASR model is trained only by the cross-entropy loss.
            ctc_zero_infinity: bool = True
                Whether to zero infinite losses and the associated gradients when calculating the CTC loss.
            att_guid_sigma: float = 0.0
                The value of the sigma used to calculate the attention guidance loss.
                If this argument is set to 0.0, the attention guidance will be disabled.

        """
        # arguments checking
        assert 0 <= ctc_weight < 1, \
            f"Your input ctc_weight must be a float number in [0, 1)! (got {ctc_weight}) " \
            f"{self.__class__.__name__} doesn't support pure CTC training."

        # initialize cross-entropy loss
        self.ce_loss = CrossEntropy(is_normalized=ce_normalized, label_smoothing=label_smoothing)

        # initialize ctc loss
        if ctc_weight > 0:
            if self.device != 'cpu' and self.tokenizer.ignore_idx != 0:
                raise RuntimeError(f"For speeding up CTC calculation by CuDNN, "
                                   f"please set the blank id to 0 (got {self.tokenizer.ignore_idx}).")
            # construct the CTC layer lazily
            if not hasattr(self, 'ctc_layer'):
                self.ctc_layer = TokenPostnet(input_size=self.encoder.output_size, vocab_size=self.tokenizer.vocab_size)
            self.ctc_loss = CTCLoss(weight=ctc_weight, blank=self.tokenizer.ignore_idx, zero_infinity=ctc_zero_infinity)

        # initialize attention guidance loss
        if att_guid_sigma != 0:
            assert 'encdec' in self.return_att_type, \
                "If you want to enable attention guidance for ASR training, please include 'encdec' in return_att_type."
            self.att_guid_loss = AttentionGuidance(sigma=att_guid_sigma)

        # initialize teacher-forcing accuracy for validation
        self.accuracy = Accuracy()

        # initialize error rate (CER & WER) for evaluation
        self.error_rate = ErrorRate(tokenizer=self.tokenizer)

    @staticmethod
    def bad_cases_selection_init_fn() -> List[List[str or int]] or None:
        return [
            ['wer', 'max', 30],
            ['cer', 'max', 30],
            ['feat_token_len_ratio', 'min', 30],
            ['feat_token_len_ratio', 'max', 30],
            ['text_confid', 'min', 30],
            ['text_confid', 'max', 30]
        ]

    def batch_preprocess_fn(self, batch_data: Dict):
        """

        Args:
            batch_data:

        Returns:

        """

        def process_strings(data_dict: Dict):
            """
            turn the text strings into tensors and get their lengths

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
                       epoch: int = None,
                       domain: str = None,
                       return_att: bool = False,
                       **kwargs) -> Dict[str, torch.Tensor]:
        """

        Args:
            feat: (batch, feat_maxlen, feat_dim)
                The input speech data. feat_dim = 1 in the case of raw speech waveforms.
            feat_len: (batch,)
                The lengths of input speech data
            text: (batch, text_maxlen)
                The input text data with <sos/eos> at the beginning and end
            text_len: (batch,)
                The lengths of input text data
            epoch: int
                The number of the current training epoch.
                Mainly used for mean&std calculation in the feature normalization
            domain: str = None
            return_att: bool
                Controls whether the attention matrices of each layer in the encoder and decoder will be returned.
            kwargs:
                Temporary register used to store the redundant arguments.

        Returns:
            A dictionary containing all the ASR model outputs necessary to calculate the losses

        """

        # para checking
        assert feat.size(0) == text.size(0) and feat_len.size(0) == text_len.size(0), \
            "The amounts of utterances and sentences are not equal to each other."
        assert feat_len.size(0) == feat.size(0), \
            "The amounts of utterances and their lengths are not equal to each other."
        assert text_len.size(0) == text.size(0), \
            "The amounts of sentences and their lengths are not equal to each other."

        # remove the <sos/eos> at the end of each sentence
        for i in range(text_len.size(0)):
            text[i, text_len[i] - 1] = self.tokenizer.ignore_idx
        text, text_len = text[:, :-1], text_len - 1

        # --- 1. Input Feature to Encoder Hidden Representation --- #
        enc_returns = self.encoder(feat=feat, feat_len=feat_len, epoch=epoch, domain=domain)
        # Transformer-based encoder additionally returns the encoder self-attention
        if len(enc_returns) == 4:
            enc_feat, enc_feat_mask, enc_attmat, enc_hidden = enc_returns
        # RNN-based encoder doesn't return any attention
        elif len(enc_returns) == 3:
            (enc_feat, enc_feat_mask, enc_hidden), enc_attmat = enc_returns, None
        else:
            raise RuntimeError

        # --- 2. Encoder Hidden Representation to Decoder Hidden Representation --- #
        dec_returns = self.decoder(enc_feat=enc_feat, enc_feat_mask=enc_feat_mask, text=text, text_len=text_len)
        # Transformer-based decoder additionally returns the decoder self-attention
        if len(dec_returns) == 4:
            dec_feat, dec_attmat, encdec_attmat, dec_hidden = dec_returns
        # RNN-based decoder only returns the encoder-decoder attention
        elif len(dec_returns) == 3:
            (dec_feat, encdec_attmat, dec_hidden), dec_attmat = dec_returns, None
        else:
            raise RuntimeError

        # initialize the asr output to be the decoder predictions
        outputs = dict(
            logits=dec_feat
        )

        # --- 3. Encoder Hidden Representation to CTC Prediction --- #
        if hasattr(self, 'ctc_layer'):
            ctc_logits = self.ctc_layer(enc_feat)
            outputs.update(
                ctc_logits=ctc_logits,
                enc_feat_len=torch.sum(enc_feat_mask.squeeze(dim=1), dim=-1)
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
                          logits: torch.Tensor,
                          text: torch.Tensor,
                          text_len: torch.Tensor,
                          ctc_logits: torch.Tensor = None,
                          enc_feat_len: torch.Tensor = None,
                          att: Dict[str, List[torch.Tensor]] = None,
                          ce_loss_fn: CrossEntropy = None,
                          ctc_loss_fn: CTCLoss = None,
                          att_guid_loss_fn: AttentionGuidance = None,
                          **kwargs) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]) or Dict[str, torch.Tensor]:
        """

        Args:
            logits:
            text:
            text_len:
            ctc_logits:
            enc_feat_len:
            att:
            ce_loss_fn:
            ctc_loss_fn:
            att_guid_loss_fn:
            **kwargs:

        Returns:

        """
        accuracy = self.accuracy(logits=logits, text=text, text_len=text_len)
        metrics = dict(accuracy=accuracy.detach())

        # the external cross-entropy loss function has the higher priority
        if ce_loss_fn is None:
            ce_loss_fn = self.ce_loss
        ce_loss = ce_loss_fn(logits=logits, text=text, text_len=text_len)

        # if ctc_loss is specified, record ctc_loss and ce_loss in the metrics Dict
        if ctc_loss_fn is not None or hasattr(self, 'ctc_loss'):
            # the external ctc loss function has the higher priority
            if ctc_loss_fn is None:
                ctc_loss_fn = self.ctc_loss

            ctc_loss = ctc_loss_fn(ctc_logits, enc_feat_len, text, text_len)
            loss = (1 - ctc_loss_fn.weight) * ce_loss + ctc_loss_fn.weight * ctc_loss
            metrics.update(ctc_loss=ctc_loss.clone().detach(), ce_loss=ce_loss.clone().detach())
        # if ctc_loss is not specified, only record the overall loss in the returned Dicts
        else:
            loss = ce_loss

        # if att_guid_loss is given, record att_guid_loss in the metrics Dict
        if att_guid_loss_fn is not None or hasattr(self, 'att_guid_loss'):
            # the external attention guidance loss function has the higher priority
            if att_guid_loss_fn is None:
                att_guid_loss_fn = self.att_guid_loss

            # layer_num * (batch, head_num, ...) -> (batch, layer_num * head_num, ...)
            att_tensor = torch.cat(att['encdec'], dim=1)
            att_guid_loss = att_guid_loss_fn(att_tensor, text_len, enc_feat_len)
            loss += att_guid_loss
            metrics.update(att_guid_loss=att_guid_loss.clone().detach())

        losses = dict(loss=loss)
        # .clone() here prevents the loss from being modified by accum_grad
        metrics.update(loss=loss.clone().detach())

        if self.training:
            return losses, metrics
        else:
            return metrics

    def matrix_snapshot(self, vis_logs: List, hypo_attention: Dict, subfolder_names: List[str] or str, epoch: int):
        """
        recursively snapshot all the attention matrices

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
                    sep_save=False, data_save=True, subfolder_names=subfolder_names
                )
            )

    def attention_reshape(self, hypo_attention: Dict, prefix_list: List = None) -> Dict:

        if prefix_list is None:
            prefix_list = []

        # process the input data by different data types
        if isinstance(hypo_attention, Dict):
            return {key: self.attention_reshape(value, prefix_list + [key]) for key, value in hypo_attention.items()}
        # count the list from the end to the beginning
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
                  feat: torch.Tensor = None,
                  feat_len: torch.Tensor = None,
                  text: torch.Tensor = None,
                  text_len: torch.Tensor = None):

        # remove the padding zeros at the end of the input feat
        for i in range(len(feat)):
            feat[i] = feat[i][:feat_len[i]]

        # visualization inference is default to be done by teacher-forcing
        if len(self.visual_infer_conf) == 0:
            self.visual_infer_conf = dict(teacher_forcing=True)
        # obtain the inference results
        infer_results = self.inference(infer_conf=self.visual_infer_conf, return_att=True,
                                       feat=feat, feat_len=feat_len, text=text, text_len=text_len)

        # --- snapshot the objective metrics --- #
        vis_logs = []
        # CER, WER, Confidence, and Length Ratio
        materials = dict()
        for metric in ['cer', 'wer', 'text_confid']:
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
            # snapshot input audio
            vis_logs.append(
                dict(
                    plot_type='audio', materials=dict(input_audio=copy.deepcopy(feat[0])),
                    sample_rate=self.sample_rate, audio_format=self.audio_format, subfolder_names=sample_index
                )
            )
            # snapshot real text
            vis_logs.append(
                dict(
                    materials=dict(real_text=[copy.deepcopy(self.tokenizer.tensor2text(text[0][1: -1]))]),
                    plot_type='text', subfolder_names=sample_index
                )
            )
        # hypothesis text
        if 'text' not in epoch_records[sample_index].keys():
            epoch_records[sample_index]['text'] = []
        epoch_records[sample_index]['text'].append(infer_results['text']['content'][0])
        # snapshot the information in the materials
        vis_logs.append(
            dict(
                materials=dict(hypo_text=copy.deepcopy(epoch_records[sample_index]['text'])),
                plot_type='text', epoch=epoch, x_stride=snapshot_interval, subfolder_names=sample_index
            )
        )

        # hypothesis attention matrix
        infer_results['att'] = self.attention_reshape(infer_results['att'])
        self.matrix_snapshot(vis_logs=vis_logs, hypo_attention=copy.deepcopy(infer_results['att']),
                             subfolder_names=sample_index, epoch=epoch)
        return vis_logs

    def inference(self,
                  infer_conf: Dict,
                  feat: torch.Tensor = None,
                  feat_len: torch.Tensor = None,
                  text: torch.Tensor = None,
                  text_len: torch.Tensor = None,
                  domain: str = None,
                  return_att: bool = False,
                  decode_only: bool = False,
                  teacher_forcing: bool = False) -> Dict[str, Any]:
        """
        The inference function for ASR models. There are two steps in this function:
            1. Decode the input speech into hypothesis transcript
            2. Evaluate the hypothesis transcript by the ground-truth

        This function can be called for model evaluation, on-the-fly model visualization, and even pseudo transcript
        generation during training.

        Args:
            # --- Testing data arguments --- #
            feat: torch.Tensor
                The speech data to be inferred.
            feat_len: torch.Tensor
                The length of `feat`.
            text: torch.Tensor
                The ground-truth transcript for the input speech
            text_len: torch.Tensor
                The length of `text`.
            # --- Explicit inference arguments --- #
            domain: str = None
                This argument indicates which domain the input speech belongs to.
                It's used to indicate the `ASREncoder` member how to encode the input speech.
            return_att: bool = False
                Whether the attention matrix of the input speech is returned.
            decode_only: bool = False
                Whether skip the evaluation step and do the decoding step only.
            teacher_forcing: bool = True
                Whether you use the teacher-forcing technique to generate the hypothesis transcript.
            # --- Implicit inference arguments given by infer_cfg from runner.py --- #
            infer_conf: Dict
                The inference configuration given from the `infer_cfg` in your `exp_cfg`.
                For more details, please refer to speechain.infer_func.beam_search.beam_searching.

        Returns: Dict
            A Dict containing all the decoding and evaluation results.

        """
        assert feat is not None and feat_len is not None

        # --- 0. Hyperparameter & Model Preparation Stage --- #
        # in-place replace infer_conf to protect the original information
        infer_conf = copy.deepcopy(infer_conf)
        if 'decode_only' in infer_conf.keys():
            decode_only = infer_conf.pop('decode_only')
        if 'teacher_forcing' in infer_conf.keys():
            teacher_forcing = infer_conf.pop('teacher_forcing')
        hypo_text, hypo_text_len, feat_token_len_ratio, hypo_text_confid, hypo_att = None, None, None, None, None

        # --- 1. The 1st Pass: ASR Decoding by Beam Searching --- #
        if not teacher_forcing:
            # copy the input data in advance for data safety
            model_input = copy.deepcopy(dict(feat=feat, feat_len=feat_len))

            # Encoding input speech
            enc_feat, enc_feat_mask, _, _ = self.encoder(domain=domain, **model_input)

            # generate the model hypothesis
            infer_results = beam_searching(enc_feat=enc_feat,
                                           enc_feat_mask=enc_feat_mask,
                                           decode_one_step=self.decoder,
                                           ctc_decode_fn=self.ctc_layer if hasattr(self, 'ctc_layer') else None,
                                           vocab_size=self.tokenizer.vocab_size,
                                           sos_eos=self.tokenizer.sos_eos_idx,
                                           padding_idx=self.tokenizer.ignore_idx,
                                           **infer_conf)
            hypo_text = infer_results['hypo_text']
            hypo_text_len = infer_results['hypo_text_len']
            feat_token_len_ratio = infer_results['feat_token_len_ratio']
            hypo_text_confid = infer_results['hypo_text_confid']

        # --- 2. The 2nd Pass: ASR Decoding by Teacher Forcing --- #
        if teacher_forcing or return_att:
            infer_results = self.module_forward(feat=feat, feat_len=feat_len,
                                                text=text if teacher_forcing else hypo_text,
                                                text_len=text_len if teacher_forcing else hypo_text_len,
                                                return_att=return_att)
            # return the attention matrices
            if return_att:
                hypo_att = infer_results['att']

            # update the hypothesis text-related data in the teacher forcing mode
            if teacher_forcing:
                # the last token is meaningless because the text is padded with eos at the end
                temperature = infer_conf['temperature'] if 'temperature' in infer_conf.keys() else 1.0
                infer_results['logits'] = torch.log_softmax(infer_results['logits'][:, :-1] / temperature, dim=-1)
                hypo_text_confid, hypo_text = torch.max(infer_results['logits'], dim=-1)
                # the original text contains both sos at the beginning and eos at the end
                hypo_text_len = text_len - 2
                feat_token_len_ratio = feat_len / hypo_text_len
                # sum up the log-probability of all time steps to get the confidence
                length_penalty = infer_conf['length_penalty'] if 'length_penalty' in infer_conf.keys() else 1.0
                hypo_text_confid = torch.sum(hypo_text_confid, dim=-1) / (hypo_text_len ** length_penalty)

        # turn the data all the unsupervised metrics into the cpu version (List)
        # consider one <sos/eos> at the end, so hypo_text_len is added to 1
        hypo_text_len, feat_token_len_ratio, hypo_text_confid = \
            to_cpu(hypo_text_len + 1), to_cpu(feat_token_len_ratio), to_cpu(hypo_text_confid)

        # --- 3. Unsupervised Metrics Calculation (ground-truth text is not involved here) --- #
        # recover the text tensors back to text strings (removing the padding and sos/eos tokens)
        hypo_text = [self.tokenizer.tensor2text(hypo[(hypo != self.tokenizer.ignore_idx) &
                                                     (hypo != self.tokenizer.sos_eos_idx)]) for hypo in hypo_text]

        # in the decoding-only mode, only the hypothesis-related results will be returned
        outputs = dict(
            text=dict(format='txt', content=hypo_text),
            text_len=dict(format='txt', content=hypo_text_len),
            feat_token_len_ratio=dict(format='txt', content=feat_token_len_ratio),
            text_confid=dict(format='txt', content=hypo_text_confid)
        )

        # add the attention matrix into the output Dict, only used for model visualization during training
        # because it will consume too much time for saving the attention matrices of all testing samples during testing
        if return_att:
            outputs.update(
                att=hypo_att
            )

        # recover the text tensors back to text strings (removing the padding and sos/eos tokens)
        text = [self.tokenizer.tensor2text(real[(real != self.tokenizer.ignore_idx) &
                                                (real != self.tokenizer.sos_eos_idx)]) for real in text]
        # evaluation reports for all the testing instances
        instance_report_dict, align_table_list, cer_list, wer_list, insertion_list, deletion_list, substitution_list = \
            {}, [], [], [], [], [], []
        # loop each sentence
        for i in range(len(text)):
            # add the confidence into instance_reports.md
            if 'Hypothesis Confidence' not in instance_report_dict.keys():
                instance_report_dict['Hypothesis Confidence'] = []
            instance_report_dict['Hypothesis Confidence'].append(f"{hypo_text_confid[i]:.6f}")

            # add the frame-token length ratio into instance_reports.md
            if 'Feature-Token Length Ratio' not in instance_report_dict.keys():
                instance_report_dict['Feature-Token Length Ratio'] = []
            instance_report_dict['Feature-Token Length Ratio'].append(f"{feat_token_len_ratio[i]:.2f}")

            # --- 4. Supervised Metrics Calculation (Reference is involved here)  --- #
            if not decode_only:
                # obtain the cer and wer metrics
                cer, wer = self.error_rate(hypo_text=hypo_text[i], real_text=text[i])
                i_num, d_num, s_num, align_table = get_word_edit_alignment(hypo_text[i], text[i])

                # record the string of hypothesis-reference alignment table
                align_table_list.append(align_table)

                # record the CER value of the current data instance
                cer_list.append(cer[0])
                if 'CER' not in instance_report_dict.keys():
                    instance_report_dict['CER'] = []
                instance_report_dict['CER'].append(f"{cer[0]:.2%}")

                # record the WER value of the current data instance
                wer_list.append(wer[0])
                if 'WER' not in instance_report_dict.keys():
                    instance_report_dict['WER'] = []
                instance_report_dict['WER'].append(f"{wer[0]:.2%}")

                # record the word insertion value of the current data instance
                insertion_list.append(i_num)
                if 'Word Insertion' not in instance_report_dict.keys():
                    instance_report_dict['Word Insertion'] = []
                instance_report_dict['Word Insertion'].append(f"{i_num}")

                # record the word deletion value of the current data instance
                deletion_list.append(d_num)
                if 'Word Deletion' not in instance_report_dict.keys():
                    instance_report_dict['Word Deletion'] = []
                instance_report_dict['Word Deletion'].append(f"{d_num}")

                # record the word substitution value of the current data instance
                substitution_list.append(s_num)
                if 'Word Substitution' not in instance_report_dict.keys():
                    instance_report_dict['Word Substitution'] = []
                instance_report_dict['Word Substitution'].append(f"{s_num}")

        # register the instance reports and the strings of alignment tables for generating instance_reports.md
        self.register_instance_reports(md_list_dict=instance_report_dict, extra_string_list=align_table_list)

        # not return the supervised metrics in the decoding-only mode
        if not decode_only:
            outputs.update(
                cer=dict(format='txt', content=cer_list),
                wer=dict(format='txt', content=wer_list),
                insertion=dict(format='txt', content=insertion_list),
                deletion=dict(format='txt', content=deletion_list),
                substitution=dict(format='txt', content=substitution_list)
            )
        return outputs


class MultiDomainARASR(ARASR):
    """

    """
    def criterion_init(self,
                       loss_weights: Dict[str, float] = None,
                       ce_loss: Dict = None,
                       ctc_loss: Dict = None,
                       att_guid_loss: Dict = None,
                       **kwargs):
        """

        Args:
            loss_weights:
            ce_loss:
            ctc_loss:
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
                nested_loss = dict()
                assert len(loss_dict) == len(self.loss_weights), ""
                for name, conf in loss_dict.items():
                    assert name in self.loss_weights.keys()
                    nested_loss[name] = loss_class(**conf)
                return nested_loss
            else:
                raise RuntimeError("Your loss configuration must be either Dict[str, Any] or Dict[str, Dict[str, Any]]")

        # cross-entropy will be initialized no matter whether ce_loss is given or not
        self.ce_loss = recur_init_loss_by_dict(ce_loss, CrossEntropy) if ce_loss is not None else CrossEntropy()
        # only initialize ctc loss if it is given
        if ctc_loss is not None:
            self.ctc_loss = recur_init_loss_by_dict(ctc_loss, CTCLoss)
        # only initialize attention-guidance loss if it is given
        if att_guid_loss is not None:
            self.att_guid_loss = recur_init_loss_by_dict(att_guid_loss, AttentionGuidance)

        # initialize teacher-forcing accuracy for validation
        self.accuracy = Accuracy()

        # initialize error rate (CER & WER) for evaluation
        self.error_rate = ErrorRate(tokenizer=self.tokenizer)

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
            return super(MultiDomainARASR, self).module_forward(**batch_data)
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
            return {domain: super(MultiDomainARASR, self).module_forward(domain=domain, **general_args, **domain_data)
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
            return super(MultiDomainARASR, self).criterion_forward(**data_output_dict)
        # Multi-dataloader scenario
        # For semi-supervised training or validation of out-domain semi-supervised ASR where we may have multiple
        # data-label pairs in a single batch, we need to go through forward function once for each pair.
        else:
            losses, metrics, domain_list = dict(), dict(), list(data_output_dict.keys())
            for domain in domain_list:
                # initialize the cross-entropy loss function
                ce_loss_fn = self.ce_loss[domain] if isinstance(self.ce_loss, Dict) else self.ce_loss
                # initialize the ctc loss function only if ctc_loss is created
                if hasattr(self, 'ctc_loss'):
                    ctc_loss_fn = self.ctc_loss[domain] if isinstance(self.ctc_loss, Dict) else self.ctc_loss
                else:
                    ctc_loss_fn = None
                # initialize the attention-guidance loss function only if att_guid_loss is created
                if hasattr(self, 'att_guid_loss'):
                    att_guid_loss_fn = self.att_guid_loss[domain] if isinstance(self.att_guid_loss, Dict) \
                        else self.att_guid_loss
                else:
                    att_guid_loss_fn = None

                # call the criterion_forward() of the parent class by the initialized loss functions
                _losses, _metrics = super(MultiDomainARASR, self).criterion_forward(
                    ce_loss_fn=ce_loss_fn, ctc_loss_fn=ctc_loss_fn, att_guid_loss_fn=att_guid_loss_fn,
                    **data_output_dict[domain])

                # update the losses and metrics Dicts by the domain name at the beginning
                losses.update(**{f"{domain}_{_key}": _value for _key, _value in _losses.items()})
                metrics.update(**{f"{domain}_{_key}": _value for _key, _value in _metrics.items()})

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
            return super(MultiDomainARASR, self).inference(infer_conf=infer_conf, **test_batch)

        # sub-Dict means that the domain information is given for ASR inference
        else:
            assert len(test_batch) == 1, \
                "If you want to evaluate the ASR model by multiple domains, please evaluate them one by one."
            for domain, domain_batch in test_batch.items():
                return super(MultiDomainARASR, self).inference(infer_conf=infer_conf, domain=domain, **domain_batch)
