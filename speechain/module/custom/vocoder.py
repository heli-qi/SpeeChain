"""
    Author: Sashi Novitasari/Andros Tjandra
    Affiliation: NAIST
    Date: 2022.08
"""
import torch
from torch.cuda.amp import autocast
from typing import Dict
from speechain.module.abs import Module
from speechain.utilbox.import_util import import_class
from speechain.utilbox.train_util import make_mask_from_len
from .cbhg1d import CBHG1d

import torch.nn as nn
from speechain.utilbox.train_util import generator_act_module




class TacotronV1Inverter(Module):
    """
    Tacotron vocoder/inverter based on CBHG + (later) Griffin-Lim
    """

    def module_init(self,vocoder:Dict) :
        """
        Args:
            vocoder: dict
                - in_size (int): input feature size
                - out_size (int): output size
                - projection_size ( List(int)): list of linear layer dimension (output side). to project cbhg output into final output
                - projection_fn (str): activation for projection layer
                - projection_do (int): dropout for projection layer
        """
        self.in_size = vocoder['in_size'] 
        self.out_size = vocoder['out_size'] 
        self.projection_size = vocoder['projection_size'] 
        self.projection_fn = vocoder['projection_fn'] 
        self.projection_do = vocoder['projection_do'] 
        
        # init CBHG layer
        cbhg_cfg={}
        self.inverter_lyr = CBHG1d(self.in_size, 
                conv_proj_filter=[256, self.in_size], **cbhg_cfg)
        _tmp = []
        prev_size = self.inverter_lyr.out_features
        for ii in range(len(self.projection_size)) :
            _tmp.append(nn.Linear(prev_size, self.projection_size[ii]))
            _tmp.append(generator_act_module(self.projection_fn))
            _tmp.append(nn.Dropout(p=self.projection_do))
            prev_size = self.projection_size[ii]
            pass
        # init linear projection layer for output
        _tmp.append(nn.Linear(prev_size, self.out_size))
        self.projection_lyr = nn.Sequential(*_tmp)
        pass


    def forward(self, feat_src, feat_src_len=None) :
        """
        Args:
            feat_src: tensor (batch, seqlen,featdim)
                input feature for model
            feat_src_len: tensor (batch)
        """
        batch, max_seq_len, _ = feat_src.size()
        # Inverter/CBHG
        res = self.inverter_lyr(feat_src, feat_src_len)
        res = res.contiguous().view(batch * max_seq_len, -1)
        
        # data projection into final output
        res = self.projection_lyr(res)
        res = res.contiguous().view(batch, max_seq_len, -1)
        return dict(pred_res=res)
        
    