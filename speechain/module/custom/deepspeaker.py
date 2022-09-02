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
import numbers

import torch.nn as nn
import torch.nn.functional as F
from speechain.utilbox.train_util import generator_act_module
from torch.autograd import Variable



"""
Deep Speaker : an End to End Neural Speaker Embedding System
Main class: DeepSpeakerCNN
Ref : https://arxiv.org/abs/1705.02304
"""

class Conv2dEv(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_size, 
            stride=1, padding=0, dilation=1, groups=1, bias=True) :
        r"""
        A guide to convolution arithmetic for deep learning
        https://arxiv.org/pdf/1603.07285v1.pdf

        padding : ['same', 'valid', 'full']
        """
        super(Conv2dEv, self).__init__()
        self.cutoff = [False, False]
        if isinstance(kernel_size, int) :
            kernel_size = (kernel_size, kernel_size)

        if isinstance(padding, numbers.Number) :
            padding = (padding, padding)
        elif padding == 'valid' :
            padding = (0, 0)
        elif padding == 'full' :
            padding = (kernel_size[0]-1, kernel_size[1]-1)
        elif padding == 'same' :
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            for ii in range(len(kernel_size)) :
                if kernel_size[ii] % 2 == 0 :
                    self.cutoff[ii] = True
        else :
            if isinstance(padding, (tuple,list)) :
                new_padding = list(padding)
                for ii, pad_ii in enumerate(padding) :
                    if isinstance(pad_ii, numbers.Number) :
                        new_padding[ii] = pad_ii
                    if pad_ii == 'valid' :
                        new_padding[ii] = 0
                    elif pad_ii == 'full' :
                        new_padding[ii] = kernel_size[ii]-1
                    elif pad_ii == 'same' :
                        new_padding[ii] = kernel_size[ii] // 2
                        if kernel_size[ii] % 2 == 0:
                            self.cutoff[ii] = True
                padding = new_padding
            else :
                raise NotImplementedError()
                        
        self.conv_lyr = nn.Conv2d(in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias)
        pass

    def forward(self, input) :
        output = self.conv_lyr(input)
        if any(self.cutoff) :
            h, w = output.size()[2:]
            if self.cutoff[0] :
                h -= 1
            if self.cutoff[1] :
                w -= 1
            output = output[:, :, 0:h, 0:w]
        return output
    pass

class ResidualBlock2D(nn.Module) :
    def __init__(self, out_channels, kernel_size=[3, 3], stride=1, 
            num_layers=2, use_bn=False, fn_act=F.leaky_relu) :
        super().__init__()
        self.out_channels = out_channels
        self.stride=stride

        self.num_layers = num_layers
        self.use_bn = use_bn
        self.fn_act = fn_act

        self.conv_lyrs = nn.ModuleList()
        if self.use_bn :
            self.bn_lyrs = nn.ModuleList()
        for ii in range(num_layers) :
            self.conv_lyrs.append(Conv2dEv(out_channels, out_channels, 
                kernel_size=kernel_size, stride=stride, padding='same'))
            if self.use_bn :
                self.bn_lyrs.append(nn.BatchNorm2d(out_channels))
        pass
    
    def forward(self, x) :
        transformed_x = x
        for ii in range(self.num_layers) :
            transformed_x = self.conv_lyrs[ii](transformed_x) 
            if ii == self.num_layers - 1 : # if ii is last layer #
                transformed_x = x + transformed_x # original x + projected x
            else :
                pass
            transformed_x = self.fn_act(transformed_x)

            # use bn after nonlinearity
            # ref : https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
            if self.use_bn :
                transformed_x = self.bn_lyrs[ii](transformed_x)
        return transformed_x


def _auto_detect_cuda(module) :
    if isinstance(module, torch.nn.Module) :
        return next(module.parameters()).is_cuda
    if isinstance(module, bool) :
        return module
    if isinstance(module, int) :
        return module >= 0
    if isinstance(module, torch.autograd.Variable) :
        return module.data.is_cuda
    if isinstance(module, torch.tensor._TensorBase) :
        return module.is_cuda
    raise NotImplementedError()

def generate_seq_mask(seq_len, max_len=None) :
    """
    seq_len : list of each batch length
    """
    batch = len(seq_len)
    max_len = max(seq_len) if max_len is None else max_len
    mask = torch.FloatTensor(batch, max_len).zero_()
    for ii in range(batch) :
        mask[ii, 0:seq_len[ii]] = 1.0
    return mask

def generator_act_fn(name) :
    if not isinstance(name, str) :
        return name
    act_fn = None
    if name is None or name.lower() in ['none', 'null'] :
        act_fn = (lambda x : x)
    else :
        try :
            act_fn = getattr(F, name)
        except AttributeError :
            act_fn = getattr(torch, name)
    return act_fn
    
class DeepSpeakerCNN(Module) :
    def module_init(self, spkrec:Dict):
        """
        Args:
            spkrec: dict
                - in_size (int): input feature size
                - out_emb_size (int): speaker embedding output size (will be used in TTS)
                - channel_size (List(int)): channel size for conv layer
                - kernel_size (List(int,int)): kernel size for conv layer
                - stride (List(int,int)): stride size for conv layer
                - conv_fn_act (str): activation function for conv layer
                - conv_do (int): dropout function for conv layer
                - linear_do (int): dropout function for output linear layer
                - pool_fn (str): type of conv pooling operation. 'avg' or 'max'
                - num_speaker (int): number of speaker class
        """

        self.in_size        = spkrec['in_size']
        self.out_emb_size   = spkrec['out_emb_size']
        self.channel_size   = spkrec['channel_size']
        self.kernel_size    = spkrec['kernel_size']
        self.stride         = spkrec['stride']
        self.conv_fn_act    = spkrec['conv_fn_act']
        self.conv_do        = spkrec['conv_do']
        self.linear_do      = spkrec['linear_do']
        self.pool_fn        = spkrec['pool_fn']
        
        # check model spec #
        assert len(self.channel_size) == len(self.kernel_size) == len(self.stride)
        assert self.pool_fn in ['avg', 'max'], "pool_fn must be 'avg' or 'max'"
        
        self.num_layers = len(self.channel_size)

        # build model #
        # build conv layer #
        self.conv_lyr = nn.ModuleList()
        self.resblock_lyr = nn.ModuleList()
        curr_in_size = 1
        for ii in range(self.num_layers) :
            self.conv_lyr.append(Conv2dEv(curr_in_size, self.channel_size[ii], 
                kernel_size=tuple(self.kernel_size[ii]), stride=tuple(self.stride[ii])))
            self.resblock_lyr.append(ResidualBlock2D(out_channels=self.channel_size[ii]))
            curr_in_size = self.channel_size[ii]

        in_emb_size = self.channel_size[-1]
        self.lin_emb_lyr = nn.Linear(in_emb_size, self.out_emb_size)

        # build softmax (for pretraining) #
        self.num_speaker = spkrec['num_speaker']
        self.lin_softmax_lyr = nn.Linear(self.out_emb_size, self.num_speaker)
        

    def forward(self, input, input_len=None) :
        """
        Args:
            input: tensor (batch, seqlen,featdim)
            input_len: tensor (batch)
        Returns:
            res: speaker embedding
        """
        batch, max_input_len, in_size = input.size()

        # apply masking #
        if input_len is not None :
            mask_input = Variable(generate_seq_mask(input_len, max_len=max_input_len).unsqueeze(-1))
            if input.is_cuda:
                mask_input =mask_input.cuda(input.device)
            input = input * mask_input

        res = input.unsqueeze(1) 
        # apply conv 
        for ii in range(self.num_layers) :
            res = self.conv_lyr[ii](res)
            res = generator_act_fn(self.conv_fn_act)(res)
            res = self.resblock_lyr[ii](res)
            if self.conv_do > 0.0 :
                res = F.dropout(res, p=self.conv_do, training=self.training)

        # res = [batch, out_channel, seq_len, n_dim] #
        # pool across seq_len #
        if self.pool_fn == 'avg' :
            res = F.avg_pool2d(res, kernel_size=[res.size(2), 1], stride=1)
        elif self.pool_fn == 'max' :
            res = F.max_pool2d(res, kernel_size=[res.size(2), 1], stride=1)
        else :
            raise ValueError("pool_fn {} is not implemented".format(self.pool_fn))

        # affine transform #
        # res = [batch, out_channel, 1, n_dim] #
        res = F.avg_pool2d(res, kernel_size=[1, res.size(-1)], stride=1)
        
        # res = [batch, out_channel, 1, 1] #
        res = res.squeeze(-1).squeeze(-1) # res = [batch, out_channel]
        res = self.lin_emb_lyr(res)
        if self.linear_do > 0.0 :
            res = F.dropout(res, p=self.linear_do, training=self.training)
        
        # normalize to unit-norm #
        res = res / torch.norm(res, p=2, dim=1, keepdim=True)
        return res
    
    def forward_softmax(self, emb) :
        """
        Extra function to calculate speaker probability (classification)
        """
        return dict(pred_res=self.lin_softmax_lyr(emb))
    
    

    pass
