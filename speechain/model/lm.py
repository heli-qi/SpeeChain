import warnings
from typing import Dict, List

import torch
import copy
import numpy as np

from speechain.model.abs import Model
from speechain.tokenizer.char import CharTokenizer
from speechain.tokenizer.sp import SentencePieceTokenizer

from speechain.module.standalone.lm import LanguageModel
from speechain.criterion.cross_entropy import CrossEntropy
from speechain.criterion.accuracy import Accuracy

from speechain.utilbox.tensor_util import to_cpu
from speechain.utilbox.train_util import text2tensor_and_len, make_mask_from_len


class LM(Model):
    """
    Auto-Regressive Attention-based Language Model.

    """

    def module_init(self,
                    token_type: str,
                    token_vocab: str,
                    emb: Dict,
                    encoder: Dict,
                    return_att_head_num: int = 2,
                    return_att_layer_num: int = 2):
        """

        Args:
            token_type:
            token_vocab:
            emb:
            encoder:
            return_att_head_num:
            return_att_layer_num:

        """
        # --- 1. Module-independent Initialization --- #
        # initialize the tokenizer
        if token_type.lower() == 'char':
            self.tokenizer = CharTokenizer(token_vocab, copy_path=self.result_path)
        elif token_type.lower() == 'sentencepiece':
            self.tokenizer = SentencePieceTokenizer(token_vocab, copy_path=self.result_path)
        else:
            raise ValueError(f"Unknown token_type {token_type}. "
                             f"Currently, {self.__class__.__name__} supports one of ['char', 'sentencepiece'].")

        self.return_att_head_num = return_att_head_num
        self.return_att_layer_num = return_att_layer_num

        # --- 2. Module Initialization --- #
        self.lm = LanguageModel(vocab_size=self.tokenizer.vocab_size, emb=emb, encoder=encoder)

    def criterion_init(self, **criterion_conf):
        """

        Args:
            **criterion_conf:

        """
        # initialize cross-entropy loss
        self.ce_loss = CrossEntropy(**criterion_conf)

        # initialize teacher-forcing accuracy for validation
        self.accuracy = Accuracy()

    @staticmethod
    def bad_cases_selection_init_fn() -> List[List[str or int]] or None:
        return [
            ['text_ppl', 'max', 30],
            ['text_ppl', 'min', 30],
            ['text_confid', 'max', 30],
            ['text_confid', 'min', 30]
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
                       text: torch.Tensor,
                       text_len: torch.Tensor,
                       epoch: int = None,
                       domain: str = None,
                       return_att: bool = False,
                       **kwargs) -> Dict:
        """

        Args:
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

        """
        assert text_len.size(0) == text.size(0), \
            "The amounts of sentences and their lengths are not equal to each other."

        # remove the <sos/eos> at the end of each sentence
        for i in range(text_len.size(0)):
            text[i, text_len[i] - 1] = self.tokenizer.ignore_idx
        text, text_len = text[:, :-1], text_len - 1

        # Next-Token prediction
        logits, _, enc_attmat = self.lm(text, text_len)

        # initialize the asr output to be the decoder predictions
        outputs = dict(
            logits=logits
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
            if enc_attmat is not None:
                outputs.update(
                    att=shrink_attention(enc_attmat)
                )
        return outputs

    def criterion_forward(self,
                          logits: torch.Tensor,
                          text: torch.Tensor,
                          text_len: torch.Tensor) -> \
            (Dict[str, torch.Tensor], Dict[str, torch.Tensor]) or Dict[str, torch.Tensor]:
        """

        Args:
            logits:
            text:
            text_len:

        Returns:

        """
        accuracy = self.accuracy(logits=logits, text=text, text_len=text_len)

        # mask generation for the input text
        text_mask = make_mask_from_len(text_len - 1, return_3d=False)
        if text.is_cuda:
            text_mask = text_mask.cuda(text.device)

        # perplexity calculation
        log_prob = torch.log_softmax(logits, dim=-1)
        text_prob = log_prob.gather(-1, text[:, 1:].view(text.size(0), -1, 1)).squeeze(dim=-1)
        text_prob = text_prob.masked_fill(~text_mask, 0.0)
        text_ppl = torch.exp(torch.sum(text_prob, dim=-1) * (- 1 / (text_len - 1))).mean()

        metrics = dict(accuracy=accuracy.detach(), text_ppl=text_ppl.clone().detach())

        loss = self.ce_loss(logits=logits, text=text, text_len=text_len)
        losses = dict(loss=loss)
        # .clone() here prevents the loss from being modified by accum_grad
        metrics.update(loss=loss.clone().detach())

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
                  text: torch.Tensor = None,
                  text_len: torch.Tensor = None):

        # visualization inference is default to be done by teacher-forcing
        if len(self.visual_infer_conf) == 0:
            self.visual_infer_conf = dict()

        # obtain the inference results
        infer_results = self.inference(infer_conf=self.visual_infer_conf, return_att=True,
                                       text=text, text_len=text_len)

        # --- snapshot the objective metrics --- #
        vis_logs = []
        # numerical metrics recording
        materials = dict()
        for metric in ['text_confid', 'text_ppl']:
            # store each target metric into materials
            if metric not in epoch_records[sample_index].keys():
                epoch_records[sample_index][metric] = []
            epoch_records[sample_index][metric].append(infer_results[metric]['content'][0])
            materials[metric] = epoch_records[sample_index][metric]
        # save the visualization log
        vis_logs.append(
            dict(
                plot_type='curve', materials=copy.deepcopy(materials), epoch=epoch,
                xlabel='epoch', x_stride=snapshot_interval, sep_save=False, subfolder_names=sample_index
            )
        )

        # --- snapshot the subjective metrics --- #
        # record the input audio and real text at the first snapshotting step
        if epoch // snapshot_interval == 1:
            # snapshot input text
            vis_logs.append(
                dict(
                    materials=dict(real_text=[copy.deepcopy(self.tokenizer.tensor2text(text[0][1: -1]))]),
                    plot_type='text', subfolder_names=sample_index
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
                  domain: str = None,
                  return_att: bool = False,) -> Dict[str, Dict[str, str or List]]:
        """

        Args:
            infer_conf:
            text:
            text_len:
            domain:
            return_att:

        Returns:

        """
        assert text is not None and text_len is not None

        # copy the input data in advance for data safety
        model_input = copy.deepcopy(dict(text=text, text_len=text_len))

        # LM Decoding by Teacher Forcing
        infer_results = self.module_forward(return_att=return_att, **model_input)
        outputs = dict()

        # add the attention matrix into the output Dict, only used for model visualization during training
        # because it will consume too much time for saving the attention matrices of all testing samples during testing
        if return_att:
            outputs.update(
                att=infer_results['att']
            )

        # --- Perplexity Calculation --- #
        # the last token (EOS) should be included for perplexity
        log_prob = torch.log_softmax(infer_results['logits'], dim=-1)
        hypo_text_prob = log_prob.gather(-1, text[:, 1:].view(text.size(0), -1, 1)).squeeze(dim=-1)
        hypo_text_ppl = torch.exp(torch.sum(hypo_text_prob, dim=-1) * (- 1 / (text_len - 1)))

        # --- Confidence Calculation --- #
        # the last token is meaningless because the text is padded with eos at the end
        log_prob = log_prob[:, :-1]
        hypo_text_prob, hypo_text = torch.max(log_prob, dim=-1)
        # the original text contains both sos at the beginning and eos at the end
        # sum up the log-probability of all time steps to get the confidence
        length_penalty = infer_conf['length_penalty'] if 'length_penalty' in infer_conf.keys() else 1.0
        hypo_text_confid = torch.sum(hypo_text_prob, dim=-1) / ((text_len - 2) ** length_penalty)

        # turn the data all the unsupervised metrics into the cpu version (List)
        hypo_text_confid, hypo_text_ppl = to_cpu(hypo_text_confid), to_cpu(hypo_text_ppl)

        # recover the text tensors back to text strings (removing the padding and sos/eos tokens)
        hypo_text = [self.tokenizer.tensor2text(hypo[(hypo != self.tokenizer.ignore_idx) &
                                                     (hypo != self.tokenizer.sos_eos_idx)]) for hypo in hypo_text]

        # in the decoding-only mode, only the hypothesis-related results will be returned
        outputs.update(
            text=dict(format='txt', content=hypo_text),
            text_confid=dict(format='txt', content=hypo_text_confid),
            text_ppl=dict(format='txt', content=hypo_text_ppl)
        )

        # evaluation reports for all the testing instances
        instance_report_dict = {}
        # loop each utterance
        for i in range(len(text)):
            if 'Text Confidence' not in instance_report_dict.keys():
                instance_report_dict['Text Confidence'] = []
            instance_report_dict['Text Confidence'].append(f"{hypo_text_confid[i]:.6f}")

            if 'Text Perplexity' not in instance_report_dict.keys():
                instance_report_dict['Text Perplexity'] = []
            instance_report_dict['Text Perplexity'].append(f"{hypo_text_ppl[i]:.4f}")
        # register the instance reports for generating instance_reports.md
        self.register_instance_reports(md_list_dict=instance_report_dict)

        return outputs
