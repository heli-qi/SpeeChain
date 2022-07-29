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
                        return_enc: bool = False,
                        aver_required_metrics: List = None):
        """
        Initialize the token dictionary for ASR evaluation, i.e. turn the token sequence into string.

        Args:
            token_type:
            token_dict:
            sample_rate:
            return_att:
            return_hidden:
            return_enc:
            aver_required_metrics:

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

        # testing metrics requried to calculate the average
        if aver_required_metrics is None:
            self.aver_required_metrics = ['cer', 'wer', 'hypo_time']


    def batch_preprocess(self, batch_data: Dict):
        """

        Args:
            batch_data:

        Returns:

        """

        def transform_text(data_dict: Dict):
            """
            turn the text strings into tensors and get their lengths

            Args:
                data_dict:

            Returns:

            """
            assert 'text' in data_dict.keys()
            if isinstance(data_dict['text'], List):
                for i in range(len(data_dict['text'])):
                    data_dict['text'][i] = self.tokenizer.text2tensor(data_dict['text'][i])

                text_len = torch.LongTensor([t.size(0) for t in data_dict['text']])
                text = torch.full((text_len.size(0), text_len.max().item()), self.tokenizer.ignore_idx,
                                  dtype=text_len.dtype)
                for i in range(text_len.size(0)):
                    text[i][:text_len[i]] = data_dict['text'][i]

                data_dict['text'] = text
                data_dict.update(text_len=text_len)
            return data_dict

        batch_keys = list(batch_data.keys())
        # if the elements are still Dict (multiple dataloaders)
        if isinstance(batch_data[batch_keys[0]], Dict):
            for key in batch_keys:
                batch_data[key] = transform_text(batch_data[key])
        # if the elements are tensors (single dataloader)
        elif isinstance(batch_data[batch_keys[0]], torch.Tensor):
            batch_data = transform_text(batch_data)
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
            epoch_records[sample_index][metric].append(infer_results[metric])
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
        epoch_records[sample_index]['hypo_text'].append(infer_results['hypo_text'])
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
                  teacher_forcing: bool = False) -> Dict[str, Any]:
        """

        Args:
            feat:
            feat_len:
            text:
            text_len:
            beam_size:
            maxlen_ratio:

        Returns:

        """
        start_time = time.time()

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
                hypo_text_prob = to_cpu(torch.sum(hypo_text_prob, dim=-1)) / (to_cpu(hypo_text_len) ** length_penalty)

        # the overall time = speech encoding time + beam searching time + string converting time
        hypo_time = time.time() - start_time

        # recover the string of the hypothesis text
        assert hypo_text.dim() <= 2, \
            f"The dimension of the hypothesis sequence generated in the testing stage must be less than 2, " \
            f"but got {hypo_text.dim()}!!"
        assert hypo_text.dim() == 2 and hypo_text.size(0) == 1, \
            f"The evaluation of ASR models can only be done one sentence a time, " \
            f"so the first dim of hypo_text must be 1!! Got {hypo_text.size()}."

        # calculate the error rate (CER & WER)
        assert text.dim() <= 2, \
            f"The dimension of the hypothesis sequence generated in the testing stage must be less than 2, " \
            f"but got {text.dim()}!!"
        assert text.dim() == 2 and text.size(0) == 1, \
            f"The evaluation of ASR models can only be done one sentence a time, " \
            f"so the first dim of text must be 1!! Got {text.size()}."
        # remove the sos and eos in the real text
        text = text[:, 1:-1]
        cer_wer = self.error_rate(hypo_text=hypo_text, real_text=text, tokenizer=self.tokenizer)
        hypo_text = self.tokenizer.tensor2text(hypo_text[0])

        outputs = dict(
            cer=cer_wer['cer'],
            wer=cer_wer['wer'],
            hypo_time=hypo_time,
            hypo_text_prob=hypo_text_prob[0],
            hypo_text=hypo_text,
            hypo_text_len=hypo_text_len
        )
        if return_att:
            outputs.update(
                hypo_att=hypo_att
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
