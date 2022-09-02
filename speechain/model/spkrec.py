"""
    Author: Sashi Novitasari
    Affiliation: NAIST
    Date: 2022.08
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


class DeepSpeaker(Model):
    """
    Speaker recognition model based on speech feature input. For speaker embedding generation in TTS.
    Architecture:
    1) Conv : output will be used as speaker embedding for TTS
    2) Softmax layer: for speaker ID prediction
    """

    def model_customize(self,
                        speaker_list: str):
        """
        Initialize the token dictionary for ASR evaluation, i.e. turn the token sequence into string.

        Args:
            speaker_list: str
                Filepath for speaker list.            

        """
        #load speaker list
        self.speaker_list = np.loadtxt(speaker_list, dtype=str, delimiter="\n") 
        
        #generate speaker ID to class ID map (class ID: according to speaker order in the list)
        self.speaker2idx = dict(map(reversed, enumerate(self.speaker_list)))
        
        
    def batch_preprocess(self, batch_data: Dict):
        """

        Args:
            batch_data:

        Returns:

        """

        def process_spk(data_dict: Dict):
            """
            turn the speaker ID strings into class ID tensors

            Args:
                data_dict:

            Returns:

            """
            assert 'speaker' in data_dict.keys()
            for i in range(len(data_dict['speaker'])):
                # map speaker ID into class ID
                data_dict['speaker'][i] = self.speaker2idx[data_dict['speaker'][i]]
            # to tensor
            data_dict['speaker']=torch.LongTensor(data_dict['speaker'])
            
            return data_dict

        batch_keys = list(batch_data.keys())
        
        # if the elements are still Dict (multiple dataloaders)
        if isinstance(batch_data[batch_keys[0]], Dict):
            for key in batch_keys:
                batch_data[key] = process_spk(batch_data[key])
        
        # if the elements are tensors (single dataloader)
        elif isinstance(batch_data[batch_keys[0]], torch.Tensor):
            batch_data = process_spk(batch_data)
        else:
            raise ValueError

        return batch_data

    def model_forward(self,
                      feat: torch.Tensor,
                      feat_len: torch.Tensor,**kwargs) -> Dict[str, torch.Tensor]:
        """

        Args:
            feat: (batch, feat_maxlen, feat_dim)
                The input speech data. feat_dim = 1 in the case of raw speech waveforms.
            feat_len: (batch,)
                The lengths of input speech data
            
        Returns:
            outputs: dict
                - logits: tensor
                    Speaker ID probability
                - embedding: tensor
                    Speaker embedding (for TTS)
        """

        # para checking
        assert feat_len.size(0) == feat.size(0), \
            "The amounts of utterances and their lengths are not equal to each other."

        #speaker embedding generation (for TTS)
        pred_pre_softmax    = self.spkrec(feat,feat_len)
        #speaker ID prediction
        pred_post_softmax   = self.spkrec.forward_softmax(pred_pre_softmax)
        
        #save output
        outputs = dict(
            logits=pred_post_softmax['pred_res'],
            embedding=pred_pre_softmax
        )

       
        return outputs

    def loss_calculation(self,
                         logits: torch.Tensor,
                         speaker: torch.Tensor,
                         **kwargs) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
        """

        Args:
            logits: Tensor (batch,outdim)
                Predicted speaker probability
            speaker: Tensor (batch)
                Speaker label
            **kwargs:

        Returns:

        """

        # generate sequence length for tensor mask generation in cross_entropy function. (all length=1)
        seq_len= torch.ones(logits.size(0),dtype=torch.int64)
        
        # speaker prediction loss
        loss = self.cross_entropy(logits=logits.unsqueeze(1), text=speaker.unsqueeze(1), text_len=seq_len)
        
        # speaker prediction accuracy
        accuracy = self.accuracy(logits=logits.unsqueeze(1), text=speaker.unsqueeze(1), text_len=seq_len)
        

        losses = dict(loss=loss)
        metrics = dict(loss=loss.detach(), accuracy=accuracy.detach())
        return losses, metrics

    def metrics_calculation(self,
                            logits: torch.Tensor,
                            speaker: torch.Tensor,
                            **kwargs) -> Dict[str, torch.Tensor]:
        """

        Args:
            logits: predicted speaker probability
            speaker: speaker label
            **kwargs:

        Returns:

        """
        seq_len= torch.ones((logits.size(0)),dtype=torch.int64)
        loss = self.cross_entropy(logits=logits.unsqueeze(1), text=speaker.unsqueeze(1), text_len=seq_len)
        accuracy = self.accuracy(logits=logits.unsqueeze(1), text=speaker.unsqueeze(1), text_len=seq_len)

    
        return dict(
            loss=loss.detach(),
            accuracy=accuracy.detach()
        )

   
    def visualize(self,
                  epoch: int,
                  sample_index: str,
                  snapshot_interval: int,
                  epoch_records: Dict,
                  feat: torch.Tensor,
                  feat_len: torch.Tensor,
                  speaker: torch.Tensor,
                  **kwargs):
        """

        Args:
            epoch:
            feat:
            feat_len:
            speaker:

        Returns:

        """
        # obtain the inference results
        infer_results = self.inference(feat=feat, feat_len=feat_len,
                                       speaker=speaker,
                                       teacher_forcing=True)

        # --- snapshot the objective metrics --- #
        vis_logs = []

        # performance
        materials = dict()
        for metric in ['loss','accuracy']:
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
        # record the target speaker label at the first snapshotting step
        if epoch // snapshot_interval == 1:
            # snapshot real text
            vis_logs.append(
                dict(
                    materials=dict(gold_speaker=infer_results['gold_speaker']['content']),
                    plot_type='text', subfolder_names=sample_index
                )
            )
        # predictied speaker ID
        if 'pred_speaker' not in epoch_records[sample_index].keys():
            epoch_records[sample_index]['pred_speaker'] = []
        epoch_records[sample_index]['pred_speaker'].append(infer_results['pred_speaker']['content'])
        
        # snapshot the information in the materials
        vis_logs.append(
            dict(
                materials=dict(pred_speaker=infer_results['pred_speaker']['content']),
                plot_type='text', epoch=epoch, x_stride=snapshot_interval,
                subfolder_names=sample_index
            )
        )
        
        return vis_logs

    def inference(self,
                  feat: torch.Tensor,
                  feat_len: torch.Tensor,
                  speaker: torch.Tensor,
                  teacher_forcing: bool = False,
                  **meta_info) -> Dict[str, Any]:
        """

        Args:
            # testing data arguments
            feat:
            feat_len:
            speaker:
            teacher_forcing: marker to calculate loss or not 

        Returns:

        """
        pred = self.model_forward(feat=feat.clone(),feat_len=feat_len.clone())
        pred_prob, pred_speaker_id = torch.max(pred['logits'], dim=-1)
        pred_speaker = self.speaker_list[to_cpu(pred_speaker_id)]

        outputs = dict(
            pred_prob = dict(format="txt",content=to_cpu(pred_prob)),
            pred_embedding = dict(format="npz",content=pred['embedding'].cpu().detach().numpy()),
            pred_speaker= dict(format="txt",content=pred_speaker),
            gold_speaker=dict(format="txt",content=self.speaker_list[to_cpu(speaker)]))

        # calculate the performance
        if teacher_forcing:
            metric = self.metrics_calculation(pred['logits'],speaker)
            outputs.update(
                loss=dict(format="txt",content=[metric['loss'].cpu().numpy()]),
                accuracy=dict(format="txt",content=[metric['accuracy'].cpu().numpy()]),
            )

        return outputs

