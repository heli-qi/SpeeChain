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
import json

from typing import Dict, Any, List

from speechain.model.abs import Model
from speechain.tokenizer.char import CharTokenizer
from speechain.infer_func.beam_search import beam_searching
from speechain.utilbox.tensor_util import to_cpu
from speechain.utilbox.import_util import import_class
from speechain.utilbox.train_util import make_mask_from_len
from speechain.utilbox.mask_util import generate_seq_mask

from speechain.infer_func.tts import greedy_search
from speechain.utilbox.helper import TacotronHelper


class TTSVocoder(Model):
    """
    TTS vocoder (speech feature --> speech waveform)

    """

    def model_customize(self,
                        aver_required_metrics: List = None,
                        speechfeat_generator: Dict = None,
                        speechfeat_cfg: str = None,
                        speechfeat_group=1
                        ):
        """
        Initialize vocoder module

        Args:
            aver_required_metrics:
            speechfeat_generator: Generate speech feature online (not supported yet)
            speechfeat_cfg: feature extraction configuration for waveform to linear spectrogram (later, will be used for griffin-lim)
            speechfeat_group: Speech feature (input and output) grouping/downsampling rate
        """
        self.speechfeat_group = speechfeat_group
        
        if speechfeat_generator is not None:
            speechfeat_generator_class = import_class('speechain.module.' + speechfeat_generator['type'])
            speechfeat_generator['conf'] = dict() if 'conf' not in speechfeat_generator.keys() else speechfeat_generator['conf']
            self.speechfeat_generator = speechfeat_generator_class(**speechfeat_generator['conf'])
        
        # testing metrics requried to calculate the average
        if aver_required_metrics is None:
            self.aver_required_metrics = ['loss','loss_feat','loss_freq']
        self.speechfeat_cfg = json.load(open(speechfeat_cfg))

        # tacotron module helper for griffin-lim
        self.helper= TacotronHelper(json.load(open(speechfeat_cfg)))
        

    def batch_preprocess(self, batch_data: Dict):
        """

        Args:
            batch_data:

        Returns:

        """
        def transform_feat(data_dict: Dict,data=None,data_len=None):
            """
            - Group/downsample the speech feature according to the grouping rate.
              e.g. Original (seq_len,feat_dim) = (100,80)
                   Transformed with speechfeat_group:4 = (25,320)

            Args:
                data_dict:
                data: data type (feat_src or feat_tgt)
            Returns:
                data_dict: 
            """
            if hasattr(self, 'speechfeat_generator'):
                # no amp operations for the frontend calculation to make sure the feature accuracy
                raise NotImplementedError
            if data is None:
                return data_dict
            
            #FEATURE GROUPING
            if self.speechfeat_group>1:
                #pad the feat 
                if data_dict[data].size(1)%self.speechfeat_group !=0:
                    _pad_len = self.speechfeat_group-((data_dict[data].size(1)%self.speechfeat_group))
                    data_dict[data] = torch.nn.functional.pad(data_dict[data],(0,0,0,_pad_len),"constant",0)
        
                #group the feat
                batch, seqlen, featdim= data_dict[data].size()
                data_dict[data]     = data_dict[data].reshape((batch, seqlen//self.speechfeat_group, featdim*self.speechfeat_group))
                data_dict[data_len] = torch.ceil(torch.div(data_dict[data_len],4)).type(torch.LongTensor)

            
            return data_dict


        batch_keys = list(batch_data.keys())
        
        # if the elements are still Dict (multiple dataloaders)
        if isinstance(batch_data[batch_keys[0]], Dict):
            for key in batch_keys:
                #transform input feature
                batch_data[key] = transform_feat(batch_data[key],data='feat_src',data_len='feat_src_len')
                
                #transform output feature
                batch_data[key] = transform_feat(batch_data[key],data='feat_tgt',data_len='feat_tgt_len')
                
        # if the elements are tensors (single dataloader)
        elif isinstance(batch_data[batch_keys[0]], torch.Tensor):
            #transform input feature
            batch_data = transform_feat(batch_data,data='feat_src',data_len='feat_src_len')

            #transform output feature
            batch_data = transform_feat(batch_data,data='feat_tgt',data_len='feat_tgt_len')
                
        else:
            raise ValueError

        return batch_data

    def model_forward(self,
                      feat_src: torch.Tensor,
                      feat_tgt: torch.Tensor,
                      feat_src_len: torch.Tensor,
                      feat_tgt_len: torch.Tensor) -> Dict[str, torch.Tensor]:
        """

        Args:
            feat_src: (batch, feat_maxlen, feat_dim)
                The input speech feature data (grouped/downsampled)
            feat_tgt: (batch, feat_maxlen, feat_dim)
                The target speech feature data (grouped/downsampled)
            
            feat_src_len: (batch,)
                The lengths of input speech data
            feat_tgt_len: (batch,)
                The lengths of target speech data

        Returns:
            A dictionary containing all the ASR model outputs necessary to calculate the losses

        """

        # para checking
        assert feat_src.size(0) == feat_tgt.size(0) and feat_src_len.size(0) == feat_tgt_len.size(0), \
            "The amounts of utterances and sentences are not equal to each other."
        assert feat_src_len.size(0) == feat_src.size(0), \
            "The amounts of utterances and their lengths are not equal to each other."
        assert feat_tgt_len.size(0) == feat_tgt.size(0), \
            "The amounts of sentences and their lengths are not equal to each other."
        
        # forward data to vocoder
        pred = self.vocoder(feat_src=feat_src,feat_src_len=feat_src_len)
        
        # initialize the vocoder output to be the decoder predictions
        outputs = dict(
            pred_feat=pred['pred_res'],

        )

        return outputs


    def loss_calculation(self,
                         pred_feat: torch.Tensor,
                         feat_tgt: torch.Tensor,
                         feat_tgt_len: torch.Tensor,
                         **kwargs) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
        """

        Args:
            pred_feat (tensor: batch, feat_maxlen, feat_dim) : Vocoder output (grouped)
            feat_tgt (tensor: batch, feat_maxlen, feat_dim) : Vocoder speech target (grouped)  
            feat_tgt_len (tensor: batch, maxlen): Vocoder speech target length (grouped)
            **kwargs:

        Returns:

        """
        
        # calculate overal feature loss 
        loss_feat = self.feat_loss(pred=pred_feat, tgt=feat_tgt, tgt_len=feat_tgt_len)
        
        # calculate feature loss for lower frequency
        loss_freq = self.feat_freq_loss(pred=pred_feat, tgt=feat_tgt, tgt_len=feat_tgt_len)
        
        # combine losses
        loss = loss_feat+loss_freq

        losses = dict(loss=loss)
        metrics = dict(loss=loss.detach(),loss_feat=loss_feat.detach(),loss_freq=loss_freq.detach())

        return losses, metrics

    def metrics_calculation(self,
                         pred_feat: torch.Tensor,
                         feat_tgt: torch.Tensor,
                         feat_tgt_len: torch.Tensor,
                         **kwargs) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
        """

        Args: (see loss_calculation)
            
        Returns:

        """
        loss_feat = self.feat_loss(pred=pred_feat, tgt=feat_tgt, tgt_len=feat_tgt_len)
        loss_freq = self.feat_freq_loss(pred=pred_feat, tgt=feat_tgt, tgt_len=feat_tgt_len)
        loss = loss_feat+loss_freq
        metrics = dict(loss=loss.detach(),loss_feat=loss_feat.detach(),loss_freq=loss_freq.detach())
        return metrics

    def matrix_snapshot(self, vis_logs: List, hypo_attention: Dict, subfolder_names: List[str] or str, epoch: int):

        """
        recursively snapshot all the attention matrices

        Args:
            hypo_attention:
            subfolder_names:

        Returns:

        """
        pass
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
        """


    def attention_reshape(self, hypo_attention: Dict, prefix_list: List = None) -> Dict:
        """

        Args:
            hypo_attention:
            prefix_list:

        """
        pass

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
        """


    def visualize(self,
                  epoch: int,
                  sample_index: str,
                  snapshot_interval: int,
                  epoch_records: Dict,
                  feat_src: torch.Tensor,
                  feat_src_len: torch.Tensor,
                  feat_tgt: torch.Tensor,
                  feat_tgt_len: torch.Tensor):
        """

        Args:
            epoch:
            feat_src: input speech feature (grouped)
            feat_src_len: (grouped)
            feat_tgt: output speech feature (grouped)
            feat_tgt_len: (grouped)

        Returns:

        """
        # obtain the inference results
        infer_results= self.inference(
                  feat_src=feat_src,
                  feat_src_len=feat_src_len,
                  feat_tgt=feat_tgt,
                  feat_tgt_len=feat_tgt_len,
                  return_att=True,**self.visual_infer_conf)


        
        # --- snapshot the objective metrics --- #
        vis_logs = []

        # performance metric and  hypothesis probability
        materials = dict()

        for metric in ['loss','loss_feat','loss_freq']:
            # store each target metric into materials
            if metric not in epoch_records[sample_index].keys():
                epoch_records[sample_index][metric] = []
            epoch_records[sample_index][metric].append(infer_results[metric]['content'])
            materials[metric] = epoch_records[sample_index][metric]
        
        # save the visualization log
        vis_logs.append(
            dict(
                plot_type='curve', materials=copy.deepcopy(materials), epoch=epoch,
                xlabel='epoch', x_stride=snapshot_interval,
                sep_save=False, subfolder_names=sample_index
            )
        )
        # save predicted audio (wav) visualization
        vis_logs.append(
                    dict(
                        plot_type='audio', materials=dict(pred_audio=copy.deepcopy(infer_results['hypo_signal_wav']['content'][0])),
                        sample_rate=infer_results['hypo_signal_wav']['sample_rate'], audio_format='wav', subfolder_names=sample_index
                    )
                )

        return vis_logs


    def inference(self,
                  feat_src: torch.Tensor,
                  feat_src_len: torch.Tensor,
                  feat_tgt: torch.Tensor,
                  feat_tgt_len: torch.Tensor,
                  return_att: bool = False,
                  teacher_forcing:bool =False) -> Dict[str, Any]:
        """

        Args:
            feat_src: input speech feature (grouped)
            feat_src_len: (grouped)
            feat_tgt: output speech feature (grouped)
            feat_tgt_len: (grouped)

        Returns:
        outputs: Dict
            - hypo_signal_wav : predicted speech signal (wav, ungrouped)
            - metrics (loss, loss_freq,loss_feat)

        """
        # mel to linear spectrogram
        pred = self.vocoder(feat_src=feat_src,feat_src_len=feat_src_len)['pred_res']

        # ungroup the predicted feature then run inverter (linear -> waveform)
        _, declen, featdim = pred.size()
        pred_reshaped =pred[0].reshape(declen*self.speechfeat_group,featdim//self.speechfeat_group).cpu().detach().numpy()
        pred_signal = self.helper.inv_spectrogram(pred_reshaped.T)
        
        outputs = dict(
            hypo_signal_wav=dict(format="wav",content=[pred_signal],sample_rate=self.speechfeat_cfg['sample_rate'])
        )
            
        if teacher_forcing or return_att:
            metrics = self.metrics_calculation(pred,feat_tgt,feat_tgt_len)
            outputs.update(
                loss=dict(format="txt",content=metrics['loss']),
                loss_feat=dict(format="txt",content=metrics['loss_feat']),
                loss_freq=dict(format="txt",content=metrics['loss_freq']))
        
        return outputs

