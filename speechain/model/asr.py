"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import copy
import numpy as np
import time
import torch
from typing import Dict, Any, List

from speechain.model.abs import Model
from speechain.tokenizer.char import CharTokenizer
from speechain.infer_func.beam_search import beam_searching
from speechain.utilbox.tensor_util import to_cpu
from speechain.utilbox.eval_util import get_word_edit_alignment
from speechain.utilbox.md_util import get_list_strings


class ASR(Model):
    """

    """

    def model_customize(self,
                        token_type: str,
                        token_dict: str,
                        sample_rate: int = 16000,
                        audio_format: str = 'wav',
                        return_att: bool = False,
                        return_hidden: bool = False,
                        return_enc: bool = False):
        """
        Initialize the token dictionary for ASR evaluation, i.e. turn the token sequence into string.

        Args:
            token_type:
            token_dict:
            sample_rate:
            return_att:
            return_hidden:
            return_enc:

        """
        # initialize the tokenizer
        if token_type == 'char':
            self.tokenizer = CharTokenizer(token_dict)
        else:
            raise NotImplementedError

        # initialize the sampling rate, mainly used for visualizing the input audio during training
        self.sample_rate = sample_rate
        self.audio_format = audio_format.lower()

        # controls whether some internal results are returned
        self.return_att = return_att
        self.return_hidden = return_hidden
        self.return_enc = return_enc

    def batch_preprocess(self, batch_data: Dict):
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
            assert 'text' in data_dict.keys() and isinstance(data_dict['text'], List)
            for i in range(len(data_dict['text'])):
                data_dict['text'][i] = self.tokenizer.text2tensor(data_dict['text'][i])
            text_len = torch.LongTensor([t.size(0) for t in data_dict['text']])
            text = torch.full((text_len.size(0), text_len.max().item()), self.tokenizer.ignore_idx,
                              dtype=text_len.dtype)
            for i in range(text_len.size(0)):
                text[i][:text_len[i]] = data_dict['text'][i]

            data_dict['text'] = text
            data_dict['text_len'] = text_len

            # --- Process the Speaker ID String --- #
            if 'speaker' in data_dict.keys():
                assert isinstance(data_dict['speaker'], List)
                # turn the speaker id strings into the trainable tensors

            return data_dict

        batch_keys = list(batch_data.keys())
        # if the elements are still Dict (multiple dataloaders)
        if isinstance(batch_data[batch_keys[0]], Dict):
            for key in batch_keys:
                batch_data[key] = process_strings(batch_data[key])
        # if the elements are tensors (single dataloader)
        elif isinstance(batch_data[batch_keys[0]], torch.Tensor):
            batch_data = process_strings(batch_data)
        else:
            raise ValueError

        return batch_data

    def model_forward(self,
                      feat: torch.Tensor,
                      text: torch.Tensor,
                      feat_len: torch.Tensor,
                      text_len: torch.Tensor,
                      return_att: bool = None,
                      return_hidden: bool = None,
                      return_enc: bool = None) -> Dict[str, torch.Tensor]:
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
        enc_outputs = self.encoder(feat, feat_len)

        # Decoding
        dec_outputs = self.decoder(enc_feat=enc_outputs['enc_feat'],
                                   enc_feat_mask=enc_outputs['enc_feat_mask'],
                                   text=text, text_len=text_len)

        # initialize the asr output to be the decoder predictions
        outputs = dict(
            logits=dec_outputs['output']
        )

        # return the attention results of either encoder or decoder if specified
        return_att = self.return_att if return_att is None else return_att
        if return_att:
            outputs.update(
                att=dict()
            )
            if 'att' in enc_outputs.keys():
                outputs['att'].update(
                    enc_att=enc_outputs['att']
                )
            if 'att' in dec_outputs.keys():
                outputs['att'].update(
                    dec_att=dec_outputs['att']
                )
            assert len(outputs['att']) > 0

        # return the internal hidden results of either encoder or decoder if specified
        return_hidden = self.return_hidden if return_hidden is None else return_hidden
        if return_hidden:
            outputs.update(
                hidden=dict()
            )
            if 'hidden' in enc_outputs.keys():
                outputs['hidden'].update(
                    enc_hidden=enc_outputs['hidden']
                )
            if 'hidden' in dec_outputs.keys():
                outputs['hidden'].update(
                    dec_hidden=dec_outputs['hidden']
                )
            assert len(outputs['hidden']) > 0

        # return the encoder outputs if specified
        return_enc = self.return_enc if return_enc is None else return_enc
        if return_enc:
            assert 'enc_feat' in enc_outputs.keys() and 'enc_feat_mask' in enc_outputs.keys()
            outputs.update(
                enc_feat=enc_outputs['enc_feat'],
                enc_feat_mask=enc_outputs['enc_feat_mask']
            )
        return outputs

    def loss_calculation(self,
                         logits: torch.Tensor,
                         text: torch.Tensor,
                         text_len: torch.Tensor,
                         **kwargs) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
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
        # it's better not to use loss=self.cross_entropy(...) in the dict because it may slow down the program
        losses = dict(loss=loss)
        metrics = dict(loss=loss.detach(), accuracy=accuracy.detach())
        return losses, metrics

    def metrics_calculation(self,
                            logits: torch.Tensor,
                            text: torch.Tensor,
                            text_len: torch.Tensor,
                            **kwargs) -> Dict[str, torch.Tensor]:
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

        return dict(
            loss=loss.detach(),
            accuracy=accuracy.detach()
        )

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
                  text_len: torch.Tensor):
        """

        Args:
            epoch:
            feat:
            feat_len:
            text:
            text_len:

        Returns:

        """
        # obtain the inference results
        infer_results = self.inference(feat=feat, feat_len=feat_len,
                                       text=text, text_len=text_len,
                                       return_att=True, **self.visual_infer_conf)

        # --- snapshot the objective metrics --- #
        vis_logs = []
        # CER, WER, hypothesis probability
        materials = dict()
        for metric in ['cer', 'wer', 'hypo_text_prob']:
            # store each target metric into materials
            if metric not in epoch_records[sample_index].keys():
                epoch_records[sample_index][metric] = []
            epoch_records[sample_index][metric].append(infer_results[metric][0])
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
        if 'hypo_text' not in epoch_records[sample_index].keys():
            epoch_records[sample_index]['hypo_text'] = []
        epoch_records[sample_index]['hypo_text'].append(infer_results['hypo_text'][0])
        # snapshot the information in the materials
        vis_logs.append(
            dict(
                materials=dict(hypo_text=copy.deepcopy(epoch_records[sample_index]['hypo_text'])),
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
                  feat: torch.Tensor,
                  feat_len: torch.Tensor,
                  text: torch.Tensor,
                  text_len: torch.Tensor,
                  beam_size: int = 1,
                  maxlen_ratio: float = 1.0,
                  length_penalty: float = 1.0,
                  sent_per_beam: int = 1,
                  return_att: bool = False,
                  decode_only: bool = False,
                  teacher_forcing: bool = False,
                  **meta_info) -> Dict[str, Any]:
        """

        Args:
            # testing data arguments
            feat:
            feat_len:
            text:
            text_len:
            # beam searching arguments
            beam_size:
            maxlen_ratio:
            length_penalty:
            sent_per_beam:
            # general inference arguments
            return_att:
            decode_only:
            teacher_forcing:

        Returns:

        """
        # go through beam searching process
        if not teacher_forcing:
            # copy the input data in advance for data safety
            model_input = copy.deepcopy(
                dict(feat=feat, feat_len=feat_len)
            )

            # Encoding input speech
            enc_outputs = self.encoder(**model_input)

            # generate the model hypothesis
            infer_results = beam_searching(enc_feat=enc_outputs['enc_feat'],
                                           enc_feat_mask=enc_outputs['enc_feat_mask'],
                                           decode_one_step=self.decoder,
                                           vocab_size=self.tokenizer.vocab_size,
                                           sos_eos=self.tokenizer.sos_eos_idx,
                                           beam_size=beam_size,
                                           maxlen_ratio=maxlen_ratio,
                                           length_penalty=length_penalty,
                                           sent_per_beam=sent_per_beam,
                                           padding_idx=self.tokenizer.ignore_idx)
            hypo_text = infer_results['hypo_text']
            hypo_text_len = infer_results['hypo_text_len']
            hypo_text_prob = infer_results['hypo_text_prob']

        # calculate the attention matrix
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
                infer_results['logits'] = torch.log_softmax(infer_results['logits'][:, :-1], dim=-1)
                hypo_text_prob, hypo_text = torch.max(infer_results['logits'], dim=-1)
                # the original text contains both sos at the beginning and eos at the end
                hypo_text_len = text_len - 2
                hypo_text_prob = torch.sum(hypo_text_prob, dim=-1) / (hypo_text_len ** length_penalty)

        # check the data
        assert hypo_text.size(0) == text.size(0), \
            f"The first dimension of text and hypo_text doesn't match! " \
            f"Got text.size(0)={text.size(0)} and hypo_text.size(0)={hypo_text.size(0)}."

        # obtain the cer and wer metrics
        cer_wer = self.error_rate(hypo_text=hypo_text, real_text=text, tokenizer=self.tokenizer)

        # recover the text tensors back to text strings (removing the padding and sos/eos tokens)
        hypo_text = [self.tokenizer.tensor2text(hypo[torch.logical_and(hypo != self.tokenizer.ignore_idx,
                                                                       hypo != self.tokenizer.sos_eos_idx)])
                     for hypo in hypo_text]
        text = [self.tokenizer.tensor2text(real[torch.logical_and(real != self.tokenizer.ignore_idx,
                                                                  real != self.tokenizer.sos_eos_idx)])
                for real in text]
        # text has sos/eos attached at the beginning and the end
        text_len -= 2

        # make sure that all the values in the returned Dict is in the form of List
        outputs = dict(
            hypo_text=hypo_text,
            hypo_text_prob=to_cpu(hypo_text_prob),
        )
        # text and text_len will not be used in the decoding-only mode
        if decode_only:
            return outputs

        # return the metrics calculated by text and text_len in the normal mode
        outputs.update(
            cer=cer_wer['cer'],
            wer=cer_wer['wer'],
            hypo_len_offset=to_cpu(hypo_text_len - text_len)
        )
        # add the attention matrix into the output Dict (mainly for model visualization during training)
        if return_att:
            outputs.update(
                hypo_att=hypo_att
            )

        # evaluation reports for all the testing samples
        sample_reports, insertion, deletion, substitution = [], [], [], []
        for i in range(len(text)):
            i_num, d_num, s_num, align_table = \
                get_word_edit_alignment(outputs['hypo_text'][i], text[i])
            md_list = get_list_strings(
                {
                    'CER': f"{outputs['cer'][i]:.2%}",
                    'WER': f"{outputs['wer'][i]:.2%}",
                    'Hypothesis Score': f"{outputs['hypo_text_prob'][i]:.6f}",
                    'Length Offset': f"{'+' if outputs['hypo_len_offset'][i] >= 0 else ''}{outputs['hypo_len_offset'][i]:d}",
                    'Word Insertion': f"{i_num}",
                    'Word Deletion': f"{d_num}",
                    'Word Substitution': f"{s_num}"
                }
            )

            sample_reports.append(
                '\n\n' +
                md_list +
                '\n' +
                align_table +
                '\n'
            )
            insertion.append(i_num)
            deletion.append(d_num)
            substitution.append(s_num)

        outputs['sample_reports.md'] = sample_reports
        outputs.update(
            word_insertion=insertion,
            word_deletion=deletion,
            word_substitution=substitution
        )
        return outputs


class SemiASR(ASR):
    """

    """

    def model_forward(self, **batch_data) -> Dict[str, Dict or torch.Tensor]:
        """

        Args:
            **batch_data:

        Returns:

        """
        # if the sub-batches are not Dict, go through the supervised training process
        batch_keys = list(batch_data.keys())
        if not isinstance(batch_data[batch_keys[0]], Dict):
            return super().model_forward(**batch_data)

        # otherwise, go through the semi-supervised training process
        outputs = dict()
        for name, sub_batch in batch_data.items():
            outputs[name] = super().model_forward(**sub_batch)

        return outputs

    def loss_calculation(self,
                         batch_data: Dict[str, Dict[str, torch.Tensor]],
                         model_outputs: Dict[str, Dict[str, torch.Tensor]],
                         **kwargs) -> Dict[str, torch.Tensor]:
        """

        Args:
            batch_data:
            model_outputs:
            **kwargs:

        Returns:

        """
        # supervised criteria calculation
        sup_criterion = self.cross_entropy['sup']['criterion']
        sup_loss = sup_criterion(pred=model_outputs['sup']['logits'],
                                 text=batch_data['sup']['text'],
                                 text_len=batch_data['sup']['text_len'])
        sup_accuracy = self.accuracy(pred=model_outputs['sup']['logits'],
                                     text=batch_data['sup']['text'],
                                     text_len=batch_data['sup']['text_len'])

        # unsupervised criteria calculation
        unsup_criterion = self.cross_entropy['unsup']['criterion']
        unsup_loss = unsup_criterion(pred=model_outputs['unsup']['logits'],
                                     text=batch_data['unsup']['text'],
                                     text_len=batch_data['unsup']['text_len'])
        unsup_accuracy = self.accuracy(pred=model_outputs['unsup']['logits'],
                                       text=batch_data['unsup']['text'],
                                       text_len=batch_data['unsup']['text_len'])

        # total loss calculation
        total_loss = sup_loss * self.cross_entropy['sup']['weight'] + \
                     unsup_loss * self.cross_entropy['unsup']['weight']

        return dict(
            total_loss=total_loss,
            sup_loss=sup_loss,
            unsup_loss=unsup_loss,
            sup_accuracy=sup_accuracy,
            unsup_accuracy=unsup_accuracy
        )
