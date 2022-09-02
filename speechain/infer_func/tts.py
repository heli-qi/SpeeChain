"""
    Author: Sashi Novitasari
    Affiliation: Nara Institute of Science and Technology (-2022)
    Date: 2022.08

    Modified from
    https://github.com/huggingface/transformers/blob/518bd02c9b71291333ef374f055a4d1ac3042654/src/transformers/generation_beam_search.py

"""
import torch
import numpy as np
        

def greedy_search(enc_text: torch.Tensor,
    enc_text_mask: torch.Tensor,
    decode_one_step,
    dec_init_input: torch.Tensor = None,
    bern_stop_thshld: float = 0.0,
    dec_max_len: int = 2000,
    feat_dim: int = 320,
    speaker_feat: torch.Tensor=None,
    ):

        pred = dict(pred_feat=[],pred_bern=[])
        pred_bern = []

        if dec_init_input is None:
            feat = torch.zeros((1,1,feat_dim), dtype=torch.float)
            if enc_text.is_cuda:
                feat = feat.cuda(enc_text.device)
        else:
            feat=dec_init_input
        
        feat_len=torch.ones(enc_text.size(0), dtype=torch.int)
        if enc_text.is_cuda:
            feat_len = feat_len.cuda(enc_text.device)

        while not pred['pred_bern'] or (pred['pred_bern'][-1]<bern_stop_thshld and len(pred['pred_bern'])< dec_max_len):
            if speaker_feat is None: #Single speaker
                curr_outputs = decode_one_step(enc_text=enc_text,
                                       enc_text_mask=enc_text_mask,
                                       feat=feat, feat_len=feat_len)
            else: #multi speaker
                # copy the original enc_text because enc_text will be modified in-place during decoding (for speaker embedding)
                enc_text_original = enc_text.clone()
                curr_outputs = decode_one_step(enc_text=enc_text,
                                       enc_text_mask=enc_text_mask,
                                       feat=feat, feat_len=feat_len,
                                       speaker_feat=speaker_feat)
                # revert back the enc_text into original value
                enc_text = enc_text_original

            pred['pred_feat'].append(curr_outputs['pred_feat'][:,-1:])
            pred['pred_bern'].append(curr_outputs['pred_bern'][:,-1,0])
            feat = torch.cat([feat,curr_outputs['pred_feat'][:,-1:]],1)
            feat_len = feat_len+1
        
        pred['pred_feat'] = feat[:,1:]# torch.stack(pred['feat'])
        pred['pred_bern'] = torch.stack(pred['pred_bern'],1)
        pred['pred_feat_len'] = [pred['pred_feat'].size(1) for i in range (enc_text.size(0))]

        return pred
        
        
