"""
    Author: Sashi Novitasari/Andros Tjandra
    Affiliation: NAIST (-2022)
    Date: 2022.08
"""
from typing import List
import torch
import math
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from torch.nn import functional as F
from torch.autograd import Variable

from speechain.module.abs import Module
from speechain.utilbox.train_util import generator_act_module

"""
CBHG module with the submodules.
Main class: CBHG1d
"""
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

def torchauto(module) :
    return torch.cuda if _auto_detect_cuda(module) else torch

def generate_seq_mask(seq_len, device=-1, max_len=None) :
    """
    seq_len : list of each batch length
    """
    batch = len(seq_len)
    max_len = max(seq_len) if max_len is None else max_len
    mask = torchauto(device).FloatTensor(batch, max_len).zero_()
    for ii in range(batch) :
        mask[ii, 0:seq_len[ii]] = 1.0
    return mask


__all__ = ['ConfigParser']
class ConfigParser(object) :
    """
    JSON-based config parser
    """

    @staticmethod
    def list_parser(obj, n=1) :
        obj = json.loads(obj) if isinstance(obj, str) else obj 
        obj = [obj for _ in range(n)] if not isinstance(obj, list) else obj 
        while len(obj) < n :
            obj = obj + [obj[-1]]
        return obj

    @staticmethod
    def item_parser(obj) :
        return json.loads(obj) if isinstance(obj, str) else obj


class MultiscaleConvNd(torch.nn.Module) :
    """
    Ref : 
    
    Tips : better set padding 'same' and no stride
    """
    def __init__(self, conv_nd, in_channels, out_channels, kernel_sizes,
            stride=1, padding='same', dilation=1, groups=1, bias=True) :
        super(MultiscaleConvNd, self).__init__()
        assert isinstance(kernel_sizes, (list, tuple))
        out_channels = ConfigParser.list_parser(out_channels, len(kernel_sizes))
        self.out_channels = sum(out_channels)
        
        self.multi_convnd = torch.nn.ModuleList()
        for ii in range(len(kernel_sizes)) :
            if conv_nd == 1 :
                _conv_nd_lyr = Conv1dEv
            elif conv_nd == 2 :
                _conv_nd_lyr = Conv2dEv
            self.multi_convnd.append(_conv_nd_lyr(in_channels, out_channels[ii], 
                kernel_sizes[ii], stride, padding, dilation, groups, bias))
        pass

    def forward(self, input) :
        result = []
        for ii in range(len(self.multi_convnd)) :
            result.append(self.multi_convnd[ii](input))
        result = torch.cat(result, 1) # combine in filter axis  #
        return result
        pass

class MultiscaleConv1d(MultiscaleConvNd) :
    def __init__(self, in_channels, out_channels, kernel_sizes,
            stride=1, padding='same', dilation=1, groups=1, bias=True) :
        super(MultiscaleConv1d, self).__init__(1, in_channels, out_channels, 
                kernel_sizes, stride, padding, dilation, groups, bias)

class MaxPool1dEv(torch.nn.Module) :
    def __init__(self, kernel_size, stride=None, padding=0, 
            dilation=1, return_indices=False, ceil_mode=False) :
        r"""
        A guide to convolution arithmetic for deep learning
        https://arxiv.org/pdf/1603.07285v1.pdf

        padding : ['same', 'valid', 'full']
        """
        super(MaxPool1dEv, self).__init__()
        self.cutoff = False
        if padding == 'valid' :
            padding = 0
        elif padding == 'full' :
            padding = kernel_size-1
        elif padding == 'same' :
            padding = kernel_size // 2
            if kernel_size % 2 == 0 :
                self.cutoff = True
            pass
        self.pool_lyr = torch.nn.MaxPool1d(kernel_size, stride=stride, 
                padding=padding, dilation=dilation, 
                return_indices=return_indices, ceil_mode=ceil_mode)
        pass

    def forward(self, input) :
        output = self.pool_lyr(input)
        if self.cutoff :
            h = output.size(2)
            if self.cutoff :
                h -= 1
            output = output[:, :, 0:h]
        return output

class Conv1dEv(torch.nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_size, 
            stride=1, padding=0, dilation=1, groups=1, bias=True) :
        r"""
        A guide to convolution arithmetic for deep learning
        https://arxiv.org/pdf/1603.07285v1.pdf

        padding : ['same', 'valid', 'full']
        """
        super(Conv1dEv, self).__init__()
        self.cutoff = False
        if padding == 'valid' :
            padding = 0 
        elif padding == 'full' :
            padding = kernel_size-1
        elif padding == 'same' :
            padding = kernel_size // 2
            if kernel_size % 2 == 0 :
                self.cutoff = True
            pass
        self.conv_lyr = torch.nn.Conv1d(in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias)
        pass

    def forward(self, input) :
        output = self.conv_lyr(input)
        if self.cutoff :
            h = output.size(2) - 1
            output = output[:, :, 0:h]
        return output
    pass

class HighwayFNN(torch.nn.Module) :
    def __init__(self, out_features, num_layers, fn_act=F.leaky_relu):
        super(HighwayFNN, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = torch.nn.ModuleList([torch.nn.Linear(out_features, out_features) for _ in range(num_layers)])
        self.gate = torch.nn.ModuleList([torch.nn.Linear(out_features, out_features) for _ in range(num_layers)])
        self.fn_act = fn_act

    def forward(self, x):
        """
        :param x: tensor with shape of [batch_size, size]
        :return: tensor with shape of [batch_size, size]
        applies sigm(x) * f(G(x)) + (1 - sigm(x)) * Q(x) transformation | G and Q is affine transformation,
        f is non-linear transformation, sigm(x) is affine transformation with sigmoid non-linearity
        and * is element-wise multiplication
        """
        
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.fn_act(self.nonlinear[layer](x))
            x = gate * nonlinear + (1 - gate) * x 

        return x

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

def generator_rnn(config) :
    mod_type = config['type'].lower()
    mod_args = config.get('args', [])
    mod_kwargs = config.get('kwargs', {})

    if mod_type == 'lstm' :
        _lyr = torch.nn.LSTM
    elif mod_type == 'gru' :
        _lyr = torch.nn.GRU
    elif mod_type == 'rnn' :
        _lyr = torch.nn.RNN
    elif mod_type == 'lstmcell' :
        _lyr = torch.nn.LSTMCell
    elif mod_type == 'grucell' :
        _lyr = torch.nn.GRUCell
    elif mod_type == 'rnncell' :
        _lyr = torch.nn.RNNCell
    else :
        raise NotImplementedError("rnn class {} is not implemented/existed".format(mod_type))
    return _lyr(*mod_args, **mod_kwargs)


class CBHG1d(torch.nn.Module) :
    """
    Ref : 
        Tacotron : Towards End-to-End Speech Sythesis
    CBHG composed based on 3 modules : 
        Convolution 1D - Highway Network - Bidirectional {GRU, LSTM}

    This is the 1st module 

    Input format : ( batch x time x input dim )
    Output format : ( batch x time x sum(conv_channels) )
    """
    def __init__(self, in_size, conv_bank_k=8, conv_bank_act='leaky_relu', conv_bank_filter=128,
            pool_size=2,
            conv_proj_k=[3, 3], conv_proj_filter=[128, 128], conv_proj_act=['leaky_relu', 'none'],
            highway_size=128, highway_lyr=4, highway_act='leaky_relu',
            rnn_cfgs={'type':'gru', 'bi':True}, rnn_sizes=[128],
            use_bn=True
            ) :
        super(CBHG1d, self).__init__()
        # conv bank multiscale #
        self.conv_bank_lyr = MultiscaleConv1d(in_size, conv_bank_filter, kernel_sizes=list(range(1, conv_bank_k+1)), padding='same')
        if use_bn :
            self.conv_bank_bn = torch.nn.BatchNorm1d(self.conv_bank_lyr.out_channels)
        self.conv_bank_act = conv_bank_act
        self.pool_lyr = MaxPool1dEv(pool_size, stride=1, padding='same')
        self.conv_proj_lyr = torch.nn.ModuleList()
        if use_bn :
            self.conv_proj_bn = torch.nn.ModuleList()
        prev_filter = self.conv_bank_lyr.out_channels
        for ii in range(len(conv_proj_k)) :
            self.conv_proj_lyr.append(Conv1dEv(prev_filter, conv_proj_filter[ii], kernel_size=conv_proj_k[ii], padding='same'))
            if use_bn :
                self.conv_proj_bn.append(torch.nn.BatchNorm1d(conv_proj_filter[ii]))
            prev_filter = conv_proj_filter[ii]
        self.conv_proj_act = conv_proj_act 
        assert prev_filter == in_size
        self.pre_highway_lyr = torch.nn.Linear(prev_filter, highway_size)
        self.highway_lyr = HighwayFNN(highway_size, highway_lyr)
        self.highway_act = highway_act
        self.use_bn = use_bn

        self.rnn_lyr = torch.nn.ModuleList()
        rnn_cfgs = ConfigParser.list_parser(rnn_cfgs, len(rnn_sizes))
        prev_size = highway_size
        for ii in range(len(rnn_sizes)) :
            _rnn_cfg = {}
            _rnn_cfg['type'] = rnn_cfgs[ii]['type']
            _rnn_cfg['args'] = [prev_size, rnn_sizes[ii], 1, True, True, 0, rnn_cfgs[ii]['bi']]
            self.rnn_lyr.append(generator_rnn(_rnn_cfg))
            prev_size = rnn_sizes[ii] * (2 if rnn_cfgs[ii]['bi'] else 1)
        self.output_size = prev_size
        self.out_features = prev_size
        pass

    def forward(self, input, seq_len=None) :

        batch, max_seq_len, ndim = input.size()
        # create mask #
        if seq_len is not None :
            mask_input = generate_seq_mask(seq_len=seq_len, device=self, max_len=max_seq_len)
            mask_input = Variable(mask_input.unsqueeze(-1)) # batch x seq_len x 1 #
            if input.is_cuda:
                mask_input = mask_input.cuda(input.device)
            mask_input_t12 = mask_input.transpose(1, 2)
        else :
            mask_input = None


        if mask_input is not None :
            input = input * mask_input

        seq_len = [max_seq_len for _ in range(batch)] if seq_len is None else seq_len

        res_ori = input.transpose(1, 2) # saved for residual connection #
        res = input.transpose(1, 2) # batch x ndim x seq_len #

        # apply multiscale conv #
        res = self.conv_bank_lyr(res)

        res = generator_act_fn(self.conv_bank_act)(res)

        if mask_input is not None :
            res = res * mask_input_t12

        if self.use_bn :
            res = self.conv_bank_bn(res)
            if mask_input is not None :
                res = res * mask_input_t12

        # apply pooling #
        res = self.pool_lyr(res)
        if mask_input is not None :
            res = res * mask_input_t12

        # apply conv proj #
        for ii in range(len(self.conv_proj_lyr)) :
            res = self.conv_proj_lyr[ii](res)
            res = generator_act_fn(self.conv_proj_act[ii])(res)
            if mask_input is not None :
                res = res * mask_input_t12
            if self.use_bn :
                res = self.conv_proj_bn[ii](res)
                if mask_input is not None :
                    res = res * mask_input_t12

        # apply residual connection #
        assert list(res.size()) == list(res_ori.size())
        res = res + res_ori
        if mask_input is not None :
            res = res * mask_input_t12
        # change shape for feedforward #
        res = res.transpose(1, 2) # batch x seq_len x ndim #
        res = res.contiguous().view(batch * max_seq_len, -1) # (batch * seq_len) x ndim #
        # apply pre highway #
        res = generator_act_fn(self.highway_act)(self.pre_highway_lyr(res))
        # apply highway #
        res = self.highway_lyr(res)
        # apply rnn #
        res = res.contiguous().view(batch, max_seq_len, -1) # batch x seq_len x ndim #
        if mask_input is not None :
            res = res * mask_input

        for ii in range(len(self.rnn_lyr)) :
            res = pack(res, seq_len.cpu(), batch_first=True)
            res = self.rnn_lyr[ii](res)[0]
            res, _ = unpack(res, batch_first=True)
            pass
        return res
