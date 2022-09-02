"""
    Author: Sashi Novitasari
    Affiliation: NAIST
    Date: 2022.08
"""
import copy
import numpy as np
import time
import torch
from torch.cuda.amp import autocast

from typing import Dict, Any, List

from speechain.model.abs import Model
from speechain.tokenizer.char import CharTokenizer
from speechain.infer_func.beam_search import beam_searching
from speechain.utilbox.tensor_util import to_cpu
from speechain.utilbox.import_util import import_class
from speechain.utilbox.train_util import make_mask_from_len
from speechain.utilbox.mask_util import generate_seq_mask

from speechain.infer_func.tts import greedy_search

import yaml
import argparse

class TTS(Model):
    """
    Encoder-decoder-based single/multi speaker TTS
    """

    def model_customize(self,
                        token_type: str,
                        token_dict: str,
                        sample_rate: int = 16000,
                        audio_format: str = 'wav',
                        return_att: bool = False,
                        return_hidden: bool = False,
                        return_enc: bool = False,
                        aver_required_metrics: List = None,
                        speechfeat_generator: Dict = None,
                        speechfeat_group: int = 1,
                        ):
        """
        Args:
            token_type:
            token_dict:
            sample_rate:
            return_att:
            return_hidden:
            return_enc:
            aver_required_metrics:
            speechfeat_generator: Generate speech feature online (not supported yet)
            speechfeat_group: Speech feature grouping/downsampling rate (to reduce decoder length)

        """
        # initialize the tokenizer
        if token_type == 'char':
            self.tokenizer = CharTokenizer(token_dict)
        else:
            raise NotImplementedError

        self.speechfeat_generator = None
        if speechfeat_generator is not None:
            speechfeat_generator_class = import_class('speechain.module.' + speechfeat_generator['type'])
            speechfeat_generator['conf'] = dict() if 'conf' not in speechfeat_generator.keys() else speechfeat_generator['conf']
            self.speechfeat_generator = speechfeat_generator_class(**speechfeat_generator['conf'])
        
        #initialize speech feature grouping/downsampling rate
        self.speechfeat_group = speechfeat_group
        

        # initialize the sampling rate, mainly used for visualizing the input audio during training
        self.sample_rate = sample_rate
        self.audio_format = audio_format.lower()

        # controls whether some internal results are returned
        self.return_att = return_att
        self.return_hidden = return_hidden
        self.return_enc = return_enc

        # testing metrics requried to calculate the average
        if aver_required_metrics is None:
            self.aver_required_metrics = ['loss_total','loss_feat', 'loss_bern', 'hypo_time']

    def batch_preprocess(self, batch_data: Dict):
        """

        Args:
            batch_data:

        Returns:
            batch_data:

        """
        def transform_text(data_dict: Dict):
            """
            turn the text strings into tensors and get their lengths

            Args:
                data_dict:

            Returns:
                data_dict:

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

        def transform_feat(data_dict: Dict):
            """
            - Group/downsample the speech feature according to the grouping rate.
              e.g. Original (seq_len,feat_dim) = (100,80)
                   Transformed with speechfeat_group:4 = (25,320)
            - Pad the start/end of speech feature sequence with zeros.

            Args:
                data_dict:
            Returns:
                data_dict:
            """
            if self.speechfeat_generator is not None:
                # no amp operations for the frontend calculation to make sure the feature accuracy
                raise NotImplementedError
            
            #FEATURE GROUPING
            if self.speechfeat_group>1:
                #pad the feat 
                if data_dict['feat'].size(1)%self.speechfeat_group !=0:
                    _pad_len = self.speechfeat_group-((data_dict['feat'].size(1)%self.speechfeat_group))
                    data_dict['feat'] = torch.nn.functional.pad(data_dict['feat'],(0,0,0,_pad_len),"constant",0)
        
                #group the feat
                batch, seqlen, featdim= data_dict['feat'].size()
                data_dict['feat']     = data_dict['feat'].reshape((batch, seqlen//self.speechfeat_group, featdim*self.speechfeat_group))
                data_dict['feat_len'] = torch.ceil(torch.div(data_dict['feat_len'],4)).type(torch.LongTensor)

            #pad feat for start and end of speech
            data_dict['feat'] = torch.nn.functional.pad(data_dict['feat'],(0,0,1,1),"constant",0)
            data_dict['feat_len'] = data_dict['feat_len']+2
            
            return data_dict

        def transform_speaker_feat(data_dict: Dict):
            """
            turn the speaker feat/embedding np array into torch tensor

            Args:
                data_dict:

            Returns:
                data_dict:

            """
            if 'speaker_feat' in data_dict.keys():
                data_dict['speaker_feat'] = torch.FloatTensor(np.array(data_dict['speaker_feat']))
                batch, featdim = data_dict['speaker_feat'].size()
                data_dict['speaker_feat'] = data_dict['speaker_feat'].reshape((batch,1,featdim))
            return data_dict



        batch_keys = list(batch_data.keys())
        # if the elements are still Dict (multiple dataloaders)
        if isinstance(batch_data[batch_keys[0]], Dict):
            for key in batch_keys:
                batch_data[key] = transform_text(batch_data[key])
                batch_data[key] = transform_feat(batch_data[key])
                batch_data[key] = transform_speaker_feat(batch_data[key])
                
                
        # if the elements are tensors (single dataloader)
        elif isinstance(batch_data[batch_keys[0]], torch.Tensor):
            batch_data = transform_text(batch_data)
            batch_data = transform_feat(batch_data)
            batch_data = transform_speaker_feat(batch_data)

        else:
            raise ValueError

        return batch_data


    def model_forward(self,
                      feat: torch.Tensor,
                      text: torch.Tensor,
                      feat_len: torch.Tensor,
                      text_len: torch.Tensor,
                      speaker_feat: torch.Tensor=None,
                      return_att: bool = None,
                      return_hidden: bool = None,
                      return_enc: bool = None) -> Dict[str, torch.Tensor]:
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
            speaker_feat: (batch,1,speaker embedding dim)
                Pre-extracted speaker embedding. (None=single speaker)

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
        
        # remove the last frame for autoregressive decoder input
        feat = feat[:,:-1]
        feat_len = feat_len-1
        
        # Encoding
        enc_outputs = self.encoder(text=text, text_len=text_len)

        # Decoding
        if speaker_feat is not None: #multispeaker
            dec_outputs = self.decoder(enc_text=enc_outputs['enc_feat'],
                                   enc_text_mask=enc_outputs['enc_feat_mask'],
                                   feat=feat, feat_len=feat_len,speaker_feat=speaker_feat)
        
        else:#single speaker
            dec_outputs = self.decoder(enc_text=enc_outputs['enc_feat'],
                                   enc_text_mask=enc_outputs['enc_feat_mask'],
                                   feat=feat, feat_len=feat_len)
        
        # initialize the TTS output to be the decoder predictions
        outputs = dict(
            pred_feat=dec_outputs['pred_feat'],
            pred_bern=dec_outputs['pred_bern']

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


    def bern_accuracy_calculation(self,pred_bern:torch.Tensor,tgt_bern:torch.Tensor):
        """
        Calculate TTS performance on speech end-flag (bern) prediction.

        Args:
            pred_bern (tensor: batch,seq_len) : end-flag predicted by TTS
            tgt_bern (tensor: batch,seq_len) : end-flag label

        Returns:
            acc_core_bernoulli_end (tensor: float): end-flag prediction accuracy
            f1score_end (tensor: float): end-flag prediction F1 score

        """
        _pred_end = (pred_bern > 0.0)
        _label_end = (tgt_bern > 0.5)

        acc_core_bernoulli_end = (_pred_end == _label_end).float().mean()
        EPS = torch.tensor(1e-5)
        if pred_bern.is_cuda:
            EPS = EPS.cuda(pred_bern.device)

        tp_end = ((_pred_end == 1) & (_label_end == 1)).sum().float()
        tn_end = ((_pred_end == 0) & (_label_end == 0)).sum().float()
        fp_end = ((_pred_end == 1) & (_label_end == 0)).sum().float()
        fn_end = ((_pred_end == 0) & (_label_end == 1)).sum().float()
        precision_end =  (tp_end/(tp_end + fp_end + EPS))
        recall_end = (tp_end/(tp_end + fn_end + EPS))
        f1score_end = (2*tp_end)/(2*tp_end + fp_end + fn_end + EPS)
        return acc_core_bernoulli_end, f1score_end

    def loss_calculation(self,
                         pred_feat: torch.Tensor,
                         pred_bern: torch.Tensor,
                         feat: torch.Tensor,
                         feat_len: torch.Tensor,
                         **kwargs) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
        """

        Args:
            pred_feat (tensor: batch,seq_len,feat_dim): TTS output (speech feature), grouped
            pred_bern (tensor: batch,seq_len,1): TTS output (end-flag)
            feat (tensor: batch,seq_len,feat_dim): Reference (speech feature), grouped and edge-padded 
            feat_len: feat len
            **kwargs:

        Returns:
            losses:
            metric:

        """
        #Reference speech feat: exclude the padding at the beginning of sequence
        feat_len = feat_len-1 
        feat = feat[:,1:]

        #generate reference for end-flag prediction
        feat_label_end = (1. - generate_seq_mask(feat_len-1,  max_len=feat.size(1)))
        if feat.is_cuda:
            feat_label_end = feat_label_end.cuda(feat.device)
        
        #metric calculation
        # speech feature prediction
        feat_loss = self.feat_loss(pred=pred_feat, tgt=feat, tgt_len=feat_len)
        
        # end-flag prediction
        pred_bern = pred_bern.squeeze(2)
        bern_loss = self.bern_loss(pred=pred_bern, tgt=feat_label_end)
        bern_accuracy, bern_f1 = self.bern_accuracy_calculation(pred_bern,feat_label_end)

        # combine speech feature loss and end-flag loss
        loss = feat_loss + bern_loss

        losses = dict(loss=loss)
        metrics = dict(loss=loss.detach(), feat_loss=feat_loss.detach(), bern_loss=bern_loss.detach(), bern_accuracy=bern_accuracy.detach(), bern_f1=bern_f1.detach())

        return losses, metrics

    def metrics_calculation(self,
                         pred_feat: torch.Tensor,
                         pred_bern: torch.Tensor,
                         feat: torch.Tensor,
                         feat_len: torch.Tensor,
                         **kwargs) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
        """

        Args:
            pred_feat (tensor: batch,seq_len,feat_dim): TTS output (speech feature), grouped
            pred_bern (tensor: batch,seq_len,1): TTS output (end-flag)
            feat (tensor: batch,seq_len,feat_dim): Reference (speech feature), grouped and edge-padded 
            feat_len: feat len
            **kwargs:

        Returns:
            losses:
            metric:
        (comments: see loss_calculation())
        """
        feat_len = feat_len-1
        feat = feat[:,1:]        

        feat_label_end = (1. - generate_seq_mask(feat_len-1,  max_len=feat.size(1)))
        if feat.is_cuda:
            feat_label_end = feat_label_end.cuda(feat.device)
        
        feat_loss = self.feat_loss(pred=pred_feat, tgt=feat, tgt_len=feat_len)
        
        pred_bern = pred_bern.squeeze(2)
        bern_loss = self.bern_loss(pred=pred_bern, tgt=feat_label_end)
        bern_accuracy, bern_f1 = self.bern_accuracy_calculation(pred_bern,feat_label_end)

        loss = feat_loss + bern_loss

        metrics = dict(loss=loss.detach(), feat_loss=feat_loss.detach(), bern_loss=bern_loss.detach(), bern_accuracy=bern_accuracy.detach(), bern_f1=bern_f1.detach())
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
                  speaker_feat: torch.Tensor=None):
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
        for metric in ['loss_total', 'feat_loss', 'bern_loss','bern_accuracy','bern_f1']:
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
            if self.speechfeat_generator is not None: #if the audio source is raw/wav
                vis_logs.append(
                    dict(
                        plot_type='audio', materials=dict(target_audio=copy.deepcopy(feat[0].cpu().numpy())),
                        sample_rate=self.sample_rate, audio_format=self.audio_format, subfolder_names=sample_index
                    )
                )
            else: #if the audio source is audio feature (mel spectrogram etc)
                seqlen,featdim=feat[0].size()
                feat_tgt_ungroup=feat[0].reshape(seqlen*self.speechfeat_group,featdim//self.speechfeat_group)
                vis_logs.append(
                dict(
                    plot_type='matrix', materials=dict(target_feat=copy.deepcopy(feat_tgt_ungroup.transpose(0,1).cpu().numpy())), epoch=epoch,
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
        

        if self.speechfeat_generator is not None: #if the generated speech is raw/wav
            vis_logs.append(
                dict(
                    materials=dict(hypo_feat=copy.deepcopy(infer_results['hypo_feat_npz']['content'][0])),
                    plot_type='audio', epoch=epoch, x_stride=snapshot_interval,
                    subfolder_names=sample_index
            ))

        else: #if the generated speech is speech feature
            vis_logs.append(
                dict(
                    plot_type='matrix', materials=dict(hypo_feat=copy.deepcopy(np.transpose(infer_results['hypo_feat_npz']['content'][0]))), epoch=epoch,
                    sep_save=False, data_save=False, subfolder_names=sample_index
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
                  dec_max_len: float = 2000,
                  bern_stop_thshld: float = 0.0,
                  return_att: bool = False,
                  teacher_forcing: bool = False,
                  feat_dim: int=320,
                  speaker_feat: torch.Tensor=None,) -> Dict[str, Any]:
        """

        Args:
            feat:
            feat_len:
            text:
            text_len:
            dec_max_len: maximum length for the predicted feature (grouped)
            bern_stop_thshld: end-flag threshold (stop decoding if end-flag bern > threshold)
            return_att
            teacher_forcing
            feat_dim: speech feature dimension (grouped)
            speaker_feat: (optional)

        Returns:
            outputs (dict):
            - hypo_feat_npz: predicted speech feature (ungrouped)

        """
        start_time = time.time()
        tokenized_text = self.tokenizer.tensor2text(text[0][1:-1])


        # go through greedy searching process
        if not teacher_forcing:
            # copy the input data in advance for data safety
            model_input = copy.deepcopy(
                dict(feat=feat, feat_len=feat_len)
            )

            # Encoding input text
            enc_outputs = self.encoder(text=text,text_len=text_len)

            infer_results = greedy_search(enc_text=enc_outputs['enc_feat'],
                                               enc_text_mask=enc_outputs['enc_feat_mask'],
                                               decode_one_step=self.decoder,
                                               speaker_feat=speaker_feat,
                                               bern_stop_thshld=bern_stop_thshld,
                                               dec_max_len=dec_max_len,
                                               feat_dim=feat_dim,
                                               )

            hypo_feat = infer_results['pred_feat']
            hypo_len = infer_results['pred_feat_len']

            _, declen, featdim = hypo_feat.size()

            outputs = dict(
                hypo_feat_npz=dict(format='npz',content=[hypo_feat[0].reshape(declen*self.speechfeat_group,featdim//self.speechfeat_group).cpu().detach().numpy()]),
                text=dict(format='txt',content=[tokenized_text]),
                hypo_feat_len=dict(format='txt',content=[hypo_len[0]*self.speechfeat_group])
            )
        
        elif teacher_forcing or return_att: #teacher forcing
            infer_results = self.model_forward(feat=feat if teacher_forcing else hypo_feat, feat_len=feat_len if teacher_forcing else hypo_len,
                                               text=text,
                                               text_len=text_len,
                                               return_att=return_att,
                                               speaker_feat=speaker_feat)

                
            # update the hypothesis text-related data in the teacher forcing mode
            if teacher_forcing:
                hypo_feat = infer_results['pred_feat']
                hypo_len = feat_len
                hypo_bern = infer_results['pred_bern']
            
                metrics = self.metrics_calculation(hypo_feat,hypo_bern,feat,feat_len)

                _, declen, featdim = hypo_feat.size()

                outputs = dict(
                    loss_total=dict(format='txt',content=[metrics['loss']]),
                    feat_loss=dict(format='txt',content=[metrics['feat_loss']]),
                    bern_loss=dict(format='txt',content=[metrics['bern_loss']]),
                    bern_accuracy=dict(format='txt',content=[metrics['bern_accuracy']]),
                    bern_f1 =dict(format='txt',content=[metrics['bern_f1']]),
                    hypo_feat_npz=dict(format='npz',content=[hypo_feat[0].reshape(declen*self.speechfeat_group,featdim//self.speechfeat_group).cpu().detach().numpy()]),
                    text=dict(format='txt',content=[tokenized_text]),
                    hypo_feat_len=dict(format='txt',content=[hypo_len[0]*self.speechfeat_group])
                )
        
        if return_att:
            hypo_att = infer_results['att']
            outputs.update(
                hypo_att=hypo_att
            )

        return outputs

