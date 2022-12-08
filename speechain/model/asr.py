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
from speechain.tokenizer.subword import SubwordTokenizer
from speechain.infer_func.beam_search import beam_searching
from speechain.utilbox.tensor_util import to_cpu
from speechain.utilbox.eval_util import get_word_edit_alignment
from speechain.utilbox.train_util import text2tensor_and_len

from speechain.module.encoder.asr import ASREncoder
from speechain.module.decoder.asr import ASRDecoder

from speechain.criterion.cross_entropy import CrossEntropy
from speechain.criterion.accuracy import Accuracy
from speechain.criterion.error_rate import ErrorRate


class ASR(Model):
    """
    Encoder-Decoder Automatic Speech Recognition (Enc-Dec ASR) implementation.

    """

    def module_init(self,
                    token_type: str,
                    token_vocab: str,
                    frontend: Dict,
                    enc_prenet: Dict,
                    encoder: Dict,
                    dec_prenet: Dict,
                    decoder: Dict,
                    normalize: Dict or bool = None,
                    specaug: Dict or bool = None,
                    sample_rate: int = 16000,
                    audio_format: str = 'wav'):
        """

        Args:
            # --- module_conf arguments --- #
            (mandatory) frontend:
                The configuration of the acoustic feature extraction frontend.
                This argument must be given since our toolkit doesn't support time-domain ASR.
            (optional) normalize:
                The configuration of the feature normalization module.
                This argument can be given in either a Dict or a bool value.
                In the case of the bool value, True means the default configuration and False means no normalization.
                If this argument is not given, there will be also no normalization.
            (optional) specaug:
                The configuration of the SpecAugment module.
                This argument can be given in either a Dict or a bool value.
                In the case of the bool value, True means the default configuration and False means no SpecAugment.
                If this argument is not given, there will be also no SpecAugment.
            (mandatory) enc_prenet:
                The configuration of the prenet in the encoder module.
                The encoder prenet embeds the input acoustic features into hidden embeddings before feeding them into
                the encoder.
            (mandatory) encoder:
                The configuration of the encoder module.
                The encoder embeds the hidden embeddings into the encoder representations at each time steps of the
                input acoustic features.
            (mandatory) dec_prenet:
                The configuration of the prenet in the decoder module.
                The decoder prenet embeds the input token ids into hidden embeddings before feeding them into
                the decoder.
            (mandatory) decoder:
                The configuration of the decoder module.
                The decoder predicts the probability of the next token at each time steps based on the token embeddings.
            # --- customize_conf arguments --- #
            (mandatory) token_type:
                The type of the built-in tokenizer.
            (mandatory) token_vocab:
                The absolute path of the vocabulary for the built-in tokenizer.
            (optional) sample_rate:
                The sampling rate of the input speech.
                Currently, it's used for acoustic feature extraction frontend initialization and tensorboard register of
                the input speech during model visualization.
                In the future, this argument will also be used to on-the-fly downsample the input speech.
            (optional) audio_format:
                The file format of the input speech.
                It's only used for tensorboard register of the input speech during model visualization.

        """

        # --- 1. Module-independent Initialization --- #
        # initialize the tokenizer
        if token_type.lower() == 'char':
            self.tokenizer = CharTokenizer(token_vocab)
        elif token_type.lower() == 'subword':
            # the subword model file is automatically selected in the same folder as the given vocab
            token_model = '/'.join(token_vocab.split('/')[:-1] + ['model'])
            self.tokenizer = SubwordTokenizer(token_vocab, token_model=token_model)
        else:
            raise NotImplementedError

        # initialize the sampling rate, mainly used for visualizing the input audio during training
        self.sample_rate = sample_rate
        self.audio_format = audio_format.lower()

        # --- 2. Module Initialization --- #
        # --- 2.1. Encoder construction --- #
        # the sampling rate will be first initialized
        if 'sr' not in frontend['conf'].keys():
            frontend['conf']['sr'] = self.sample_rate
        else:
            if frontend['conf']['sr'] != self.sample_rate:
                warnings.warn(
                    "The sampling rate in your frontend configuration doesn't match the one in customize_conf. "
                    "The one in your configuration will be used to extract acoustic features.")

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
        if 'vocab_size' in dec_prenet['conf'].keys():
            if dec_prenet['conf']['vocab_size'] != self.tokenizer.vocab_size:
                warnings.warn(f"Your input vocabulary size is different from the one obtained from the built-in "
                              f"tokenizer ({self.tokenizer.vocab_size}). The latter one will be used to initialize the "
                              f"decoder for correctness.")
            dec_prenet['conf'].pop('vocab_size')
        self.decoder = ASRDecoder(
            vocab_size=self.tokenizer.vocab_size,
            prenet=dec_prenet,
            decoder=decoder
        )

    def criterion_init(self, is_normalized: bool = False, label_smoothing: float = 0.0):
        """

        Args:
            is_normalized:
            label_smoothing:

        Returns:

        """
        # training loss
        self.cross_entropy = CrossEntropy(is_normalized=is_normalized, label_smoothing=label_smoothing)
        # validation metrics
        self.accuracy = Accuracy()
        self.error_rate = ErrorRate(tokenizer=self.tokenizer)

    @staticmethod
    def bad_cases_selection_init_fn() -> List[List[str or int]] or None:
        return [
            ['wer', 'max', 30],
            ['cer', 'max', 30],
            ['len_ratio', 'min', 30],
            ['len_ratio', 'max', 30],
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
                       return_hidden: bool = False,
                       return_enc: bool = False,
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
            return_hidden: bool
                Controls whether the hidden representations of each layer in the encoder and decoder will be returned.
            return_enc: bool
                Controls whether the final encoder representations will be returned.
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

        # Encoding
        enc_feat, enc_feat_mask, enc_attmat, enc_hidden = self.encoder(
            feat=feat, feat_len=feat_len, epoch=epoch, domain=domain)

        # Decoding
        dec_feat, dec_attmat, encdec_attmat, dec_hidden = self.decoder(
            enc_feat=enc_feat, enc_feat_mask=enc_feat_mask, text=text, text_len=text_len)

        # initialize the asr output to be the decoder predictions
        outputs = dict(
            logits=dec_feat
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
                enc_feat=enc_feat,
                enc_feat_mask=enc_feat_mask
            )
        return outputs

    def criterion_forward(self,
                          logits: torch.Tensor,
                          text: torch.Tensor,
                          text_len: torch.Tensor,
                          **kwargs) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]) or Dict[str, torch.Tensor]:
        """

        Args:
            logits:
            text:
            text_len:
            **kwargs:

        Returns:

        """
        loss = self.cross_entropy(logits=logits, text=text, text_len=text_len)
        accuracy = self.accuracy(logits=logits, text=text, text_len=text_len)

        # the loss and accuracy must be calculated before being assigned to the returned dict
        # it's better not to use dict(loss=self.cross_entropy(...)) in the dict because it may slow down the program
        losses = dict(loss=loss)
        # .clone() here prevents the loss from being modified by accum_grad
        metrics = dict(loss=loss.clone().detach(), accuracy=accuracy.detach())

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
                  **meta_info):
        """

        Args:
            epoch:
            feat:
            feat_len:
            text:
            text_len:

        Returns:

        """
        # remove the padding zeros at the end of the input feat
        feat = feat[:, :feat_len]

        # obtain the inference results
        infer_results = self.inference(infer_conf=self.visual_infer_conf,
                                       feat=feat, feat_len=feat_len,
                                       text=text, text_len=text_len,
                                       return_att=True)

        # --- snapshot the objective metrics --- #
        vis_logs = []
        # CER, WER, hypothesis probability
        materials = dict()
        for metric in ['cer', 'wer', 'sent_prob', 'len_offset']:
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
        if 'sent' not in epoch_records[sample_index].keys():
            epoch_records[sample_index]['sent'] = []
        epoch_records[sample_index]['sent'].append(infer_results['sent']['content'][0])
        # snapshot the information in the materials
        vis_logs.append(
            dict(
                materials=dict(hypo_text=copy.deepcopy(epoch_records[sample_index]['sent'])),
                plot_type='text', epoch=epoch, x_stride=snapshot_interval,
                subfolder_names=sample_index
            )
        )

        # hypothesis attention matrix
        infer_results['hypo_att'] = self.attention_reshape(infer_results['hypo_att'])
        self.matrix_snapshot(vis_logs=vis_logs, hypo_attention=copy.deepcopy(infer_results['hypo_att']),
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

        Args:
            # --- Testing data arguments --- #
            feat:
            feat_len:
            text:
            text_len:
            domain:
            # --- Explicit inference arguments --- #
            return_att:
            decode_only:
            teacher_forcing:
            # --- Implicit inference arguments given by infer_cfg from runner.py --- #
            infer_conf:

        """
        assert feat is not None and feat_len is not None

        # --- 0. Hyperparameter & Model Preparation Stage --- #
        # in-place replace infer_conf to protect the original information
        infer_conf = copy.deepcopy(infer_conf)
        if 'decode_only' in infer_conf.keys():
            decode_only = infer_conf.pop('decode_only')
        if 'teacher_forcing' in infer_conf.keys():
            teacher_forcing = infer_conf.pop('teacher_forcing')
        hypo_text, hypo_text_len, hypo_len_ratio, hypo_text_confid, hypo_att = None, None, None, None, None

        # --- 1. The 1st Pass: ASR Decoding by Beam Searching --- #
        if not teacher_forcing:
            # copy the input data in advance for data safety
            model_input = copy.deepcopy(
                dict(feat=feat, feat_len=feat_len)
            )

            # Encoding input speech
            enc_feat, enc_feat_mask, _, _ = self.encoder(domain=domain, **model_input)

            # generate the model hypothesis
            infer_results = beam_searching(enc_feat=enc_feat,
                                           enc_feat_mask=enc_feat_mask,
                                           decode_one_step=self.decoder,
                                           vocab_size=self.tokenizer.vocab_size,
                                           sos_eos=self.tokenizer.sos_eos_idx,
                                           padding_idx=self.tokenizer.ignore_idx,
                                           **infer_conf)
            hypo_text = infer_results['hypo_text']
            hypo_text_len = infer_results['hypo_text_len']
            hypo_len_ratio = infer_results['hypo_len_ratio']
            hypo_text_confid = infer_results['hypo_text_confid']

        # --- 2. The 2nd Pass: ASR Decoding by Teacher Forcing --- #
        if teacher_forcing or return_att:
            infer_results = self.model_forward(feat=feat, feat_len=feat_len,
                                               text=text if teacher_forcing else hypo_text,
                                               text_len=text_len if teacher_forcing else hypo_text_len,
                                               return_att=return_att)
            # return the attention matrices
            if return_att:
                hypo_att = infer_results['att']

            # update the hypothesis text-related data in the teacher forcing mode
            if teacher_forcing:
                # the last token is meant to be eos which should not appear in the hypothesis text
                infer_results['logits'] = torch.log_softmax(infer_results['logits'][:, :-1] / infer_conf['temperature'],
                                                            dim=-1)
                hypo_text_confid, hypo_text = torch.max(infer_results['logits'], dim=-1)
                # the original text contains both sos at the beginning and eos at the end
                hypo_text_len = text_len - 2
                hypo_len_ratio = torch.ones_like(hypo_text_len)
                hypo_text_confid = torch.sum(hypo_text_confid, dim=-1) / (hypo_text_len ** infer_conf['length_penalty'])

        # turn the data all the unsupervised metrics into the cpu version (List)
        # consider one <sos/eos> at the end, so hypo_text_len is added to 1
        hypo_text_len, hypo_len_ratio, hypo_text_confid = \
            to_cpu(hypo_text_len + 1), to_cpu(hypo_len_ratio), to_cpu(hypo_text_confid)

        # --- 3. Unsupervised Metrics Calculation (ground-truth text is not involved here) --- #
        # recover the text tensors back to text strings (removing the padding and sos/eos tokens)
        hypo_text = [self.tokenizer.tensor2text(hypo[(hypo != self.tokenizer.ignore_idx) &
                                                     (hypo != self.tokenizer.sos_eos_idx)]) for hypo in hypo_text]

        # in the decoding-only mode, only the hypothesis-related results will be returned
        outputs = dict(
            text=dict(format='txt', content=hypo_text),
            text_len=dict(format='txt', content=hypo_text_len),
            len_ratio=dict(format='txt', content=hypo_len_ratio),
            text_confid=dict(format='txt', content=hypo_text_confid)
        )

        # add the attention matrix into the output Dict, only used for model visualization during training
        # because it will consume too much time for saving the attention matrices of all testing samples during testing
        if return_att:
            outputs.update(
                hypo_att=hypo_att
            )

        # recover the text tensors back to text strings (removing the padding and sos/eos tokens)
        text = [self.tokenizer.tensor2text(real[(real != self.tokenizer.ignore_idx) &
                                                (real != self.tokenizer.sos_eos_idx)]) for real in text]
        # evaluation reports for all the testing samples
        instance_report_dict, align_table_list, cer_list, wer_list, insertion_list, deletion_list, substitution_list = \
            {}, [], [], [], [], [], []
        # loop each sentence
        for i in range(len(text)):
            # add the confidence into instance_reports.md
            if 'Hypothesis Confidence' not in instance_report_dict.keys():
                instance_report_dict['Hypothesis Confidence'] = []
            instance_report_dict['Hypothesis Confidence'].append(f"{hypo_text_confid[i]:.6f}")

            # add the frame-token length ratio into instance_reports.md
            if 'Length Ratio' not in instance_report_dict.keys():
                instance_report_dict['Length Ratio'] = []
            instance_report_dict['Length Ratio'].append(f"{hypo_len_ratio[i]:.2f}")

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


class SemiASR(ASR):
    """

    """

    def model_construction(self,
                           token_type: str,
                           token_vocab: str,
                           frontend: Dict,
                           enc_prenet: Dict,
                           encoder: Dict,
                           dec_prenet: Dict,
                           decoder: Dict,
                           normalize: Dict or bool = None,
                           specaug: Dict or bool = None,
                           cross_entropy: Dict = None,
                           sample_rate: int = 16000,
                           audio_format: str = 'wav',
                           **loss_weights):
        # call the model construction function of the original ASR class to build the main body of the model
        super(SemiASR, self).model_construction(token_type=token_type,
                                                token_vocab=token_vocab,
                                                frontend=frontend,
                                                enc_prenet=enc_prenet,
                                                encoder=encoder,
                                                dec_prenet=dec_prenet,
                                                decoder=decoder,
                                                normalize=normalize,
                                                specaug=specaug,
                                                cross_entropy=cross_entropy,
                                                sample_rate=sample_rate,
                                                audio_format=audio_format)
        # implicitly register all the
        for key, value in loss_weights.items():
            self.__setattr__(key, value)

    def model_forward(self, **batch_data) -> Dict[str, Dict or torch.Tensor]:
        """

        Args:
            **batch_data:

        Returns:

        """
        multi_flag = sum([not isinstance(value, torch.Tensor) for value in batch_data.values()]) == len(batch_data)

        # Single-dataloader scenario
        # probably for the validation stage of in-domain semi-supervised ASR where we only have one data-label pair
        if not multi_flag:
            return super(SemiASR, self).model_forward(**batch_data)
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
            return {key: super(SemiASR, self).model_forward(domain=key, **general_args, **value)
                    for key, value in batch_data.items()}

    def loss_calculation(self, **data_output_dict) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
        """

        Args:
            **data_output_dict:

        Returns:

        """
        multi_flag = sum([isinstance(value, Dict) for value in data_output_dict.values()]) == len(data_output_dict)

        # Single-dataloader scenario
        # probably for the validation stage of in-domain semi-supervised ASR where we only have one data-label pair
        if not multi_flag:
            losses, metrics = super(SemiASR, self).loss_calculation(**data_output_dict)
        # Multi-dataloader scenario
        # For semi-supervised training or validation of out-domain semi-supervised ASR where we may have multiple
        # data-label pairs in a single batch, we need to go through forward function once for each pair.
        else:
            losses, metrics, return_scalars = dict(), dict(), True
            for key, value in data_output_dict.items():
                # only return the trainable scalars at the first time of calling
                _losses, _metrics = super(SemiASR, self).loss_calculation(
                    domain=key, return_scalars=return_scalars, **value
                )
                return_scalars &= False

                # update the losses and metrics of each data-label pair to the result dict
                losses.update(
                    **{f"{key}_{_key}": _value for _key, _value in _losses.items()}
                )
                metrics.update(
                    **{f"{key}_{_key}": _value for _key, _value in _metrics.items()}
                )

            # normalize the combination of losses by their number, default loss weight is 1
            losses.update(
                loss=sum(
                    [value * self.__getattribute__(f"{key.split('_')[0]}_weight")
                     if hasattr(self, f"{key.split('_')[0]}_weight") else value for key, value in losses.items()]
                ) / sum(
                    [self.__getattribute__(f"{key.split('_')[0]}_weight")
                     if hasattr(self, f"{key.split('_')[0]}_weight") else 1 for key in losses.keys()]
                )
            )
        return losses, metrics

    def metrics_calculation(self, **data_output_dict) -> Dict[str, torch.Tensor]:
        """

        Args:
            **data_output_dict:

        Returns:

        """
        _, metrics = self.loss_calculation(**data_output_dict)
        return metrics

    def inference(self,
                  infer_conf: Dict,
                  **test_batch) -> Dict[str, Any]:
        """

        Args:
            infer_conf:
            **test_batch:

        Returns:

        """
        domain_flag = sum([isinstance(value, Dict) for value in test_batch.values()]) == len(test_batch)
        # no sub-Dict means one normal supervised dataloader, go through the inference function of ASR
        if not domain_flag:
            return super(SemiASR, self).inference(
                infer_conf=infer_conf, **test_batch
            )
        # sub-Dict means that the domain information is given for ASR inference
        else:
            assert len(test_batch) == 1
            for key, value in test_batch.items():
                return super(SemiASR, self).inference(
                    infer_conf=infer_conf, domain=key, **value
                )
