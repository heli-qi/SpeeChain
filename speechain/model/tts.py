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
import numpy as np
import torch

from typing import Dict, Any, List

from speechain.model.abs import Model
from speechain.tokenizer.char import CharTokenizer
from speechain.utilbox.train_util import make_mask_from_len
from speechain.utilbox.md_util import get_list_strings

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

    def model_construction(self,
                           token_type: str,
                           token_vocab: str,
                           frontend: Dict,
                           enc_emb: Dict,
                           encoder: Dict,
                           dec_prenet: Dict,
                           decoder: Dict,
                           enc_prenet: Dict = None,
                           normalize: Dict = None,
                           dec_postnet: Dict = None,
                           vocoder: Dict = None,
                           feat_loss: Dict = None,
                           stop_loss: Dict = None,
                           spk_list: str = None,
                           spk_emb: Dict = None,
                           sample_rate: int = 16000,
                           audio_format: str = 'wav',
                           reduction_factor: int = 1,
                           stop_threshold: float = 0.5):
        """
        Args:
            # --- module_conf arguments --- #
            (mandatory) frontend:
                The configuration of the acoustic feature extraction frontend.
                This argument must be given since our toolkit doesn't support time-domain TTS.
            (mandatory) enc_emb:
                The configuration of the token embedding layer in the encoder module.
                The encoder prenet embeds the input token id into token embeddings before feeding them into
                the encoder.
            (optional) enc_prenet:
                The configuration of the prenet in the encoder module.
                The encoder prenet embeds the input token embeddings into high-level embeddings before feeding them into
                the encoder.
            (mandatory) encoder:
                The configuration of the encoder module.
                The encoder embeds the input embeddings into the encoder representations at each time steps of the
                input acoustic features.
            # --- criterion_conf arguments --- #
            (optional) feat_loss:
            (optional) stop_loss:
            # --- customize_conf arguments --- #
            (mandatory) token_type:
                The type of the built-in tokenizer.
            (mandatory) token_vocab:
                The absolute path of the vocabulary for the built-in tokenizer.
            (conditionally mandatory) spk_list:
                The absolute path of the speaker list that contains all the speaker ids.
                If you would like to train a close-set multi-speaker TTS, you need to give a spk_list.
            (conditionally mandatory) spk_emb:
            (optional) sample_rate:
                The sampling rate of the target speech.
                Currently it's used for acoustic feature extraction frontend initialization and tensorboard register of
                the input speech during model visualization.
                In the future, this argument will also be used to dynamically downsample the input speech during training.
            (optional) audio_format:
                The file format of the input speech.
                It's only used for tensorboard register of the input speech during model visualization.
            (optional) reduction_factor:
                The factor that controls how much the length of output speech feature is reduced.
            (optional) stop_threshold:
                The threshold that controls whether the speech synthesis stops or not.
        """
        # --- Model-Customized Part Initialization --- #
        # initialize the tokenizer
        if token_type == 'char':
            self.tokenizer = CharTokenizer(token_vocab)
        else:
            raise NotImplementedError

        # initialize the speaker list if given
        if spk_list is not None:
            spk_list = np.loadtxt(spk_list, dtype=str)
            # when the input file is idx2spk, only retain the column of speaker ids
            if len(spk_list.shape) == 2:
                assert spk_list.shape[1] == 2
                spk_list = spk_list[:, 1]
            # otherwise, the input file must be spk_list which is a single-column file and each row is a speaker id
            elif len(spk_list.shape) != 1:
                raise RuntimeError
            # 1. remove redundant elements; 2. sort up the speaker ids in order
            spk_list = sorted(set(spk_list.tolist()))
            # 3. get the corresponding indices (start from 1 since 0 is reserved for unknown speakers)
            self.idx2spk = dict(zip(range(1, len(spk_list) + 1), spk_list))
            # 4. exchange the positions of indices and speaker ids
            self.spk2idx = dict(map(reversed, zip(range(1, len(spk_list) + 1), spk_list)))

        # initialize the sampling rate, mainly used for visualizing the input audio during training
        self.sample_rate = sample_rate
        self.audio_format = audio_format.lower()
        self.reduction_factor = reduction_factor
        self.stop_threshold = stop_threshold

        # default values of TTS topn bad case selection
        self.bad_cases_selection = [
            ['feat_token_len_ratio', 'max', 30],
            ['feat_token_len_ratio', 'min', 30],
            ['wav_len_ratio', 'max', 30],
            ['wav_len_ratio', 'min', 30]
        ]


        # --- Module Part Construction --- #
        # Encoder construction, the vocabulary size will be first initialized
        if 'vocab_size' in enc_emb['conf'].keys():
            assert enc_emb['conf']['vocab_size'] == self.tokenizer.vocab_size, \
                f"The vocab_size values are different in enc_emb and self.tokenizer! " \
                f"Got enc_emb['conf']['vocab_size']={enc_emb['conf']['vocab_size']} and " \
                f"self.tokenizer.vocab_size={self.tokenizer.vocab_size}"
        self.encoder = TTSEncoder(
            vocab_size=self.tokenizer.vocab_size,
            embedding=enc_emb,
            prenet=enc_prenet,
            encoder=encoder
        )

        # Decoder construction
        # check the sampling rate of the decoder frontend
        if 'sr' not in frontend['conf'].keys():
            frontend['conf']['sr'] = self.sample_rate
        else:
            assert frontend['conf']['sr'] == self.sample_rate, \
                "The sampling rate in your frontend configuration doesn't match the one in customize_conf!"

        # check the speaker embedding configuration
        if spk_emb is not None:
            # multi-speaker embedding mode
            spk_emb_mode = 'open' if spk_list is None else 'close'
            if 'spk_emb_mode' in spk_emb.keys():
                assert spk_emb['spk_emb_mode'] == spk_emb_mode, \
                    "Your input spk_emb_mode is different from the one generated by codes. " \
                    "It's probably because you give spk_list in the open-set mode or " \
                    "forget to give spk_list in the close-set mode."
            else:
                spk_emb['spk_emb_mode'] = spk_emb_mode

            # speaker number for the close-set multi-speaker TTS
            if spk_emb['spk_emb_mode'] == 'close':
                if 'spk_num' in spk_emb.keys():
                    # all seen speakers plus an unknown speaker
                    assert spk_emb['spk_num'] == len(self.spk2idx) + 1, \
                        "Your input spk_emb_mode is different from the one generated by codes. " \
                        "It's probably because you give spk_list in the open-set mode or " \
                        "forget to give spk_list in the close-set mode."
                else:
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
        if vocoder is None:
            self.vocoder = self.decoder.spec_to_wav
        else:
            raise NotImplementedError("Neural vocoders are not supported yet/(ToT)/~~")


        # --- Criterion Part Initialization --- #
        # training loss
        self.feat_loss = LeastError(**feat_loss)
        self.stop_loss = BCELogits(**stop_loss)
        # validation metrics
        self.stop_accuracy = Accuracy()
        self.stop_f2 = FBetaScore(beta=2.0)


    def batch_preprocess(self, batch_data: Dict):
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
            if 'spk_ids' in data_dict.keys() and hasattr(self, 'spk2idx'):
                assert isinstance(data_dict['spk_ids'], List)
                # turn the speaker id strings into the trainable tensors
                data_dict['spk_ids'] = torch.LongTensor([self.spk2idx[spk] if spk in self.spk2idx.keys()
                                                         else 0 for spk in data_dict['spk_ids']])
            # data_dict['speaker'] still remains in data_dict as original metainfo
            return data_dict

        # check whether the batch_data is made by multiple dataloaders
        multi_flag = sum([isinstance(value, Dict) for value in batch_data.values()]) == len(batch_data)
        return process_strings(batch_data) if not multi_flag else \
            {key: process_strings(value) for key, value in batch_data.items()}


    def model_forward(self,
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
        enc_outputs = self.encoder(text=text, text_len=text_len)

        # Decoding
        dec_outputs = self.decoder(enc_text=enc_outputs['enc_feat'],
                                   enc_text_mask=enc_outputs['enc_feat_mask'],
                                   feat=feat, feat_len=feat_len,
                                   spk_feat=spk_feat, spk_ids=spk_ids,
                                   epoch=epoch)
        
        # initialize the TTS output to be the decoder predictions
        outputs = dict(
            pred_stop=dec_outputs['pred_stop'],
            pred_feat_before=dec_outputs['pred_feat_before'],
            pred_feat_after=dec_outputs['pred_feat_after'],
            tgt_feat=dec_outputs['tgt_feat'],
            tgt_feat_len=dec_outputs['tgt_feat_len']
        )

        # return the attention results of either encoder or decoder if specified
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
        if return_enc:
            assert 'enc_feat' in enc_outputs.keys() and 'enc_feat_mask' in enc_outputs.keys()
            outputs.update(
                enc_feat=enc_outputs['enc_feat'],
                enc_feat_mask=enc_outputs['enc_feat_mask']
            )
        return outputs


    def loss_calculation(self,
                         pred_stop: torch.Tensor,
                         pred_feat_before: torch.Tensor,
                         pred_feat_after: torch.Tensor,
                         tgt_feat: torch.Tensor,
                         tgt_feat_len: torch.Tensor,
                         **kwargs) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
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
        stop_f2 = self.stop_f2(pred_stop_hard, tgt_stop, tgt_feat_len)

        losses = dict(loss=loss)
        # .clone() here prevents the trainable variables from value modification
        metrics = dict(loss=loss.clone().detach(),
                       feat_loss_before=feat_loss_before.clone().detach(),
                       feat_loss_after=feat_loss_after.clone().detach(),
                       stop_loss=stop_loss.clone().detach(),
                       stop_accuracy=stop_accuracy.detach(),
                       stop_f2=stop_f2.detach())

        # get the values of trainable scalars from the encoder
        encoder_scalars = self.encoder.get_trainable_scalars()
        if encoder_scalars is not None:
            metrics.update(**{'enc_' + key: value.clone().detach() for key, value in encoder_scalars.items()})
        # get the values of trainable scalars from the decoder
        decoder_scalars = self.decoder.get_trainable_scalars()
        if decoder_scalars is not None:
            metrics.update(**{'dec_' + key: value.clone().detach() for key, value in decoder_scalars.items()})

        return losses, metrics


    def metrics_calculation(self,
                            pred_stop: torch.Tensor,
                            pred_feat_before: torch.Tensor,
                            pred_feat_after: torch.Tensor,
                            tgt_feat: torch.Tensor,
                            tgt_feat_len: torch.Tensor,
                            **kwargs) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
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
        _, metrics = self.loss_calculation(
            pred_stop=pred_stop,
            pred_feat_before=pred_feat_before,
            pred_feat_after=pred_feat_after,
            tgt_feat=tgt_feat,
            tgt_feat_len=tgt_feat_len
        )
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
                  infer_conf: Dict,
                  text: torch.Tensor = None,
                  text_len: torch.Tensor = None,
                  feat: torch.Tensor = None,
                  feat_len: torch.Tensor = None,
                  spk_ids: torch.Tensor = None,
                  spk_feat: torch.Tensor = None,
                  aver_spk: bool = False,
                  feat_only: bool = False,
                  return_att: bool = False,
                  decode_only: bool = False,
                  use_dropout: bool = False,
                  vocode_only: bool = False,
                  teacher_forcing: bool = False,
                  **meta_info) -> Dict[str, Any]:
        """

        Args:
            # --- Testing data arguments --- #
            (mandatory) feat: (batch_size, feat_maxlen, feat_dim)
                Ground-truth (waveforms or acoustic features)
                Used for teacher-forcing decoding and objective evaluation
            (mandatory) feat_len: (batch_size,)
                The length of the ground-truth
            (mandatory) text: (batch_size, text_maxlen)
                The input text to be decoded
            (mandatory) text_len: (batch_size,)
                The length of the input text
            (optional) spk_ids: (batch_size,)
            (optional) spk_feat: (batch_size,)
            (optional) meta_info: Dict[str, List[str]]
            # --- General inference arguments --- #
            (optional) aver_spk:
            (optional) return_att:
            (optional) feat_only:
            (optional) decode_only:
            (optional) use_dropout:
            (optional) vocode_only:
            (optional) teacher_forcing:
            # --- TTS decoding arguments --- #
            (mandatory) infer_conf:

        """
        # --- Hyperparameter & Model Preparation Stage --- #
        # in-place replace infer_conf with its copy to protect the original information
        infer_conf = copy.deepcopy(infer_conf)
        # The following arguments should not be passed to auto_regression()
        if 'decode_only' in infer_conf.keys():
            decode_only = infer_conf.pop('decode_only')
        if 'vocode_only' in infer_conf.keys():
            vocode_only = infer_conf.pop('vocode_only')
        if 'teacher_forcing' in infer_conf.keys():
            teacher_forcing = infer_conf.pop('teacher_forcing')
        if 'use_dropout' in infer_conf.keys():
            use_dropout = infer_conf.pop('use_dropout')
        if 'aver_spk' in infer_conf.keys():
            aver_spk = infer_conf.pop('aver_spk')
        # 'feat_only' and 'stop_threshold' are kept as the arguments of auto_regression()
        # feat_only in infer_conf has the higher priority than the default values
        if 'feat_only' in infer_conf.keys():
            feat_only = infer_conf['feat_only']
        else:
            infer_conf['feat_only'] = feat_only
        # stop_threshold in infer_conf has the higher priority than the built-in one of the model
        if 'stop_threshold' not in infer_conf.keys():
            infer_conf['stop_threshold'] = self.stop_threshold
        hypo_feat, hypo_feat_len, hypo_len_ratio, hypo_wav, hypo_wav_len, hypo_wav_len_ratio, hypo_att = \
            None, None, None, None, None, None, None

        # turn the dropout layer in the decoder on for introducing variability to the synthetic utterances
        if use_dropout:
            self.decoder.turn_on_dropout()

        # set the speaker embedding to zero vectors for multi-speaker TTS
        if aver_spk and hasattr(self.decoder, 'spk_emb'):
            spk_feat = torch.zeros((text.size(0), self.decoder.spk_emb.spk_emb_dim), device=text.device)
            spk_ids = None

        # initialize the output dictionary
        outputs = dict()
        # Generate the acoustic features from the given text and then do the vocoding operation
        if not vocode_only:
            # --- The 1st Pass: TTS Auto-Regressive Decoding --- #
            if not teacher_forcing:
                # copy the input data in advance for data safety
                model_input = copy.deepcopy(
                    dict(text=text, text_len=text_len)
                )

                # Encoding input text
                enc_outputs = self.encoder(**model_input)

                # Generate the synthetic acoustic features auto-regressively
                infer_results = auto_regression(enc_text=enc_outputs['enc_feat'],
                                                enc_text_mask=enc_outputs['enc_feat_mask'],
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


            # --- The 2nd Pass: TTS Teacher-Forcing Decoding --- #
            if teacher_forcing or return_att:
                infer_results = self.model_forward(feat=feat if teacher_forcing else hypo_feat,
                                                   feat_len=feat_len if teacher_forcing else hypo_feat_len,
                                                   text=text, text_len=text_len,
                                                   spk_feat=spk_feat, spk_ids=spk_ids,
                                                   return_att=return_att)

                # return the attention matrices
                if return_att:
                    hypo_att = infer_results['att']

                # update the hypothesis text-related data in the teacher forcing mode
                if teacher_forcing:
                    outputs['feat'] = infer_results['pred_feat']
                    hypo_len = feat_len
                    hypo_bern = infer_results['pred_bern']

                    metrics = self.metrics_calculation(hypo_feat,hypo_bern,feat,feat_len)

                    _, declen, featdim = hypo_feat.size()

            # The 3rd Pass: denormalize the acoustic feature if needed
            if hasattr(self.decoder, 'normalize'):
                hypo_feat = self.decoder.normalize.recover(hypo_feat, group_ids=spk_ids)

        # Skip the generation of acoustic features and directly do the vocoding operation for the given features
        else:
            hypo_feat, hypo_feat_len = feat, feat_len


        # --- Post-processing for the Generated Acoustic Features --- #
        # recover the waveform from the acoustic feature by the vocoder
        if not feat_only:
            assert self.vocoder is not None, \
                "Please specify a vocoder if you want to recover the waveform from the acoustic features."
            hypo_wav, hypo_wav_len = self.vocoder(feat=hypo_feat, feat_len=hypo_feat_len)
            hypo_wav = [hypo_wav[i][:hypo_wav_len[i]] for i in range(len(hypo_wav))]
            outputs.update(
                wav=dict(format=self.audio_format, sample_rate=self.sample_rate, content=to_cpu(hypo_wav, tgt='numpy')),
                wav_len=dict(format='txt', content=to_cpu(hypo_wav_len))
            )

        # no features are returned in the vocoding-only mode
        if not vocode_only:
            # remove the redundant silence parts at the end of the synthetic frames
            hypo_feat = [hypo_feat[i][:hypo_feat_len[i]] for i in range(len(hypo_feat))]
            outputs.update(
                feat=dict(format='npz', content=to_cpu(hypo_feat, tgt='numpy')),
                feat_len=dict(format='txt', content=to_cpu(hypo_feat_len))
            )

        # record the speaker ID used as the reference
        if spk_ids is not None:
            assert hasattr(self, 'idx2spk')
            outputs.update(
                ref_spk=dict(format='txt', content=[self.idx2spk[s_id.item()] if s_id != 0 else 0 for s_id in spk_ids])
            )

        # add the attention matrix into the output Dict, only used for model visualization during training
        # because it will consume too much time for saving the attention matrices of all testing samples during testing
        if return_att:
            outputs.update(
                hypo_att=hypo_att
            )


        # --- Supervised Metrics Calculation (Ground-Truth is involved here) --- #
        # hypo_wav_len_ratio is a supervised metrics calculated by the decoded waveforms
        if not decode_only:
            # For the acoustic feature ground-truth, since the transformation from feature length to waveform length
            # is linear, acoustic length ratio is equal to waveform length ratio.
            hypo_wav_len_ratio = hypo_feat_len / feat_len if feat.size(-1) != 1 else hypo_wav_len / feat_len
            outputs.update(wav_len_ratio=dict(format='txt', content=to_cpu(hypo_wav_len_ratio)))


        # --- Final Report Generation Stage --- #
        # No report will be printed in the vocoding-only scenario
        if not vocode_only:
            # produce the sample report by a .md file
            sample_reports = []
            for i in range(len(hypo_feat)):
                # initialize the current report string by the unsupervised metrics
                _curr_report = '\n\n' + get_list_strings(
                    {
                        'Feature-Token Length Ratio': f"{hypo_len_ratio[i]:.2f}"
                    }
                )

                # udpate the supervised metrics into the current sample report
                if not decode_only:
                    _curr_report += get_list_strings(
                        {
                            'Hypo-Real Waveform Length Ratio': f"{hypo_wav_len_ratio[i]:.2%} "
                                                               f"({'-' if hypo_wav_len_ratio[i] < 1 else '+'}"
                                                               f"{abs(hypo_wav_len_ratio[i] - 1):.2%})"
                        }
                    )

                # update the current report with a line break at the end
                sample_reports.append(_curr_report + '\n')

            # For both decoding-only mode or normal evaluation mode, sample_reports.md will be given
            outputs['sample_reports.md'] = dict(format='txt', content=sample_reports)

        return outputs



class ReferenceSpeakerTTS(TTS):
    """

    """
    def inference(self,
                  infer_conf: Dict,
                  **test_batch) -> Dict[str, Any]:
        """

        Args:
            infer_conf:
            **test_batch:

        Returns:

        """
        ref_flag = sum([isinstance(value, Dict) for value in test_batch.values()]) == len(test_batch)
        # no sub-Dict means one normal supervised dataloader, go through the inference function of TTS
        if not ref_flag:
            return super(ReferenceSpeakerTTS, self).inference(
                infer_conf=infer_conf, **test_batch
            )
        # sub-Dict means that the reference speaker information is given for TTS inference
        else:
            assert len(test_batch) in [1, 2]
            src, ref = None, None
            for value in test_batch.values():
                if 'text' in value.keys() and 'text_len' in value.keys():
                    src = value
                else:
                    ref = value
            assert src is not None

            # the reference speaker loader is given
            if ref is not None:
                assert ('spk_ids' in ref.keys()) or ('spk_feat' in ref.keys())
                if 'spk_ids' in ref.keys():
                    assert ref['spk_ids'].size(0) in [1, src['text'].size(0)]
                    if ref['spk_ids'].size(0) < src['text'].size(0):
                        ref['spk_ids'] = ref['spk_ids'].expand(src['text'].size(0))
                else:
                    assert ref['spk_feat'].size(0) in [1, src['text'].size(0)]
                    if ref['spk_feat'].size(0) < src['text'].size(0):
                        ref['spk_feat'] = ref['spk_feat'].expand(src['text'].size(0), -1)

                return super(ReferenceSpeakerTTS, self).inference(
                    infer_conf=infer_conf,
                    feat=src['feat'] if 'feat' in src.keys() else None,
                    feat_len=src['feat_len'] if 'feat_len' in src.keys() else None,
                    text=src['text'], text_len=src['text_len'],
                    spk_ids=ref['spk_ids'] if 'spk_ids' in ref.keys() else None,
                    spk_feat=ref['spk_feat'] if 'spk_feat' in ref.keys() else None,
                )
            # no reference speaker loader is given
            else:
                return super(ReferenceSpeakerTTS, self).inference(
                    infer_conf=infer_conf, **src
                )
