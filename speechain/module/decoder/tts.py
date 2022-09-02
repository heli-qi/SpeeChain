"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch

from typing import Dict
from speechain.module.abs import Module
from speechain.utilbox.import_util import import_class
from speechain.utilbox.train_util import make_mask_from_len

class TTSDecoderSingleSpeaker(Module):
    """

    """
    def module_init(self, prenet: Dict, decoder: Dict, postnet: Dict, bernnet: Dict,featnet: Dict):
        """
        Autoregressive decoder
        Architecture:
            [input] --> prenet --> decoder --> postnet  --> featnet --> [speech feature]
                                                       |
                                                        --> bernnet --> [speech end-flag]
        Args:
            prenet: 
            decoder:
            postnet:
            bernnet: for speech end-flag prediction
            featnet: for speech feature prediction

        Returns:

        """
        _prev_output_size = None

        # Initialize prenet
        prenet_class = import_class('speechain.module.' + prenet['type'])
        prenet['conf'] = dict() if 'conf' not in prenet.keys() else prenet['conf']
        self.prenet = prenet_class(**prenet['conf'])
        _prev_output_size = self.prenet.output_size

        # Initialize decoder
        decoder_class = import_class('speechain.module.' + decoder['type'])
        decoder['conf'] = dict() if 'conf' not in decoder.keys() else decoder['conf']
        self.decoder = decoder_class(input_size=_prev_output_size, **decoder['conf'])
        _prev_output_size = self.decoder.output_size

        # Initialize postnet
        postnet_class = import_class('speechain.module.' + postnet['type'])
        postnet['conf'] = dict() if 'conf' not in postnet.keys() else postnet['conf']
        self.postnet = postnet_class(feat_dim=_prev_output_size, **postnet['conf'])
        
        # Initialize featnet
        _prev_output_size = self.postnet.output_size
        featnet_class = import_class('speechain.module.' + featnet['type'])
        featnet['conf'] = dict() if 'conf' not in featnet.keys() else featnet['conf']
        self.featnet = featnet_class(feat_dim=_prev_output_size, **featnet['conf'])

        # Initialize bernnet
        _prev_output_size = self.featnet.output_size+self.decoder.output_size
        bernnet_class = import_class('speechain.module.' + bernnet['type'])
        bernnet['conf'] = dict() if 'conf' not in bernnet.keys() else bernnet['conf']
        self.bernnet = bernnet_class(feat_dim=_prev_output_size, **bernnet['conf'])

        

    def forward(self,
               enc_text: torch.Tensor,
               enc_text_mask: torch.Tensor,
               feat: torch.Tensor,
               feat_len: torch.Tensor):
        """

        Args:
            enc_text: (torch size: batch, maxlen, enctextdim)
            enc_text_mask: (torch size: batch, 1, enctextdim)
            feat: (torch size: batch, maxlen, featdim)
            feat_len: (torch size: batch)

        Returns:

        """
        # Text Embedding
        feat,feat_len = self.prenet(feat,feat_len)

        # mask generation for the input text
        feat_mask = make_mask_from_len(feat_len)
        if feat.is_cuda:
            feat_mask = feat_mask.cuda(feat.device)

        # Decoding
        dec_results = self.decoder(src=enc_text, src_mask=enc_text_mask,
                                   tgt=feat, tgt_mask=feat_mask)

        dec_post_preout = self.postnet(dec_results['output'],feat_len)
        
        #Speech feat prediction
        dec_post_feat = self.featnet(dec_post_preout,feat_len)
        
        #End of speech prediction
        dec_post_bern = torch.cat([dec_post_feat,dec_results['output']],dim=2)
        dec_post_bern = self.bernnet(dec_post_bern,feat_len)

        # initialize the decoder outputs
        dec_outputs = dict(
            pred_feat=dec_post_feat,
            pred_bern=dec_post_bern
        )
        # if the build-in decoder has the attention results
        if 'att' in dec_results.keys():
            dec_outputs.update(
                att=dec_results['att']
            )
        # if the build-in decoder has the hidden results
        if 'hidden' in dec_results.keys():
            dec_outputs.update(
                hidden=dec_results['hidden']
            )
        return dec_outputs


class TTSDecoderMultiSpeaker(Module):
    """

    """
    def module_init(self, prenet: Dict, decoder: Dict, postnet: Dict, bernnet: Dict,featnet: Dict,spkproj_prenet: Dict,spkproj_enc: Dict,spkproj_dec: Dict):
        """
        Autoregressive decoder
        Architecture:
            1) Speaker embedding projection
                [input: speaker embedding] --> spkproj_prenet --> spkproj_enc --> [speaker_feat_enc]
                                                              |   
                                                              --> spkproj_dec --> [speaker_feat_dec]

            2) TTS speech feature synthesis
                                                    [enc_out + speaker_feat_enc]---
                                                                                  |
                                                                                  v
                [input:    feat]  --> prenet  --> [feat + speaker_feat_dec] --> decoder --> postnet  --> featnet --> [speech feature]
                                                                                                       |
                                                                                                       --> bernnet --> [speech end-flag]
        Args:
            prenet: 
            decoder:
            postnet:
            bernnet: for speech end-flag prediction
            featnet: for speech feature prediction
            spkproj_prenet: for speaker projection
            spkproj_enc:for speaker projection (encoder)
            spkproj_decfor speaker projection (decoder)


        Returns:

        """


        #---------------Initialize spkproj components
        # spkproj_prenet: for preprocessing
        spkproj_prenet_class = import_class('speechain.module.' + spkproj_prenet['type'])
        spkproj_prenet['conf'] = dict() if 'conf' not in spkproj_prenet.keys() else spkproj_prenet['conf']
        self.spkproj_prenet = spkproj_prenet_class(**spkproj_prenet['conf'])

        # spkproj_enc: for integration with encoder output
        spkproj_enc_class = import_class('speechain.module.' + spkproj_enc['type'])
        spkproj_enc['conf'] = dict() if 'conf' not in spkproj_enc.keys() else spkproj_enc['conf']
        self.spkproj_enc = spkproj_enc_class(**spkproj_enc['conf'])

        # spkproj_dec: for integration with decoder input feat
        spkproj_dec_class = import_class('speechain.module.' + spkproj_dec['type'])
        spkproj_dec['conf'] = dict() if 'conf' not in spkproj_dec.keys() else spkproj_dec['conf']
        self.spkproj_dec = spkproj_dec_class(**spkproj_dec['conf'])

        # activation for enc/dec speaker projection
        self.spkproj_act = torch.nn.Softsign()

        
        #---------------------------------------

        _prev_output_size = None
        # Initialize prenet
        prenet_class = import_class('speechain.module.' + prenet['type'])
        prenet['conf'] = dict() if 'conf' not in prenet.keys() else prenet['conf']
        self.prenet = prenet_class(**prenet['conf'])
        _prev_output_size = self.prenet.output_size

        # Initialize decoder
        decoder_class = import_class('speechain.module.' + decoder['type'])
        decoder['conf'] = dict() if 'conf' not in decoder.keys() else decoder['conf']
        self.decoder = decoder_class(input_size=_prev_output_size, **decoder['conf'])
        _prev_output_size = self.decoder.output_size

        # Initialize postnet
        postnet_class = import_class('speechain.module.' + postnet['type'])
        postnet['conf'] = dict() if 'conf' not in postnet.keys() else postnet['conf']
        self.postnet = postnet_class(feat_dim=_prev_output_size, **postnet['conf'])
        
        # Initialize featnet
        _prev_output_size = self.postnet.output_size
        featnet_class = import_class('speechain.module.' + featnet['type'])
        featnet['conf'] = dict() if 'conf' not in featnet.keys() else featnet['conf']
        self.featnet = featnet_class(feat_dim=_prev_output_size, **featnet['conf'])

        # Initialize bernnet
        _prev_output_size = self.featnet.output_size+self.decoder.output_size
        bernnet_class = import_class('speechain.module.' + bernnet['type'])
        bernnet['conf'] = dict() if 'conf' not in bernnet.keys() else bernnet['conf']
        self.bernnet = bernnet_class(feat_dim=_prev_output_size, **bernnet['conf'])

        

    def forward(self,
               enc_text: torch.Tensor,
               enc_text_mask: torch.Tensor,
               feat: torch.Tensor,
               feat_len: torch.Tensor,
               speaker_feat: torch.Tensor):
        """

        Args:
            enc_text: (torch size: batch, maxlen, enctext dim)
            enc_text_mask: (torch size: batch, 1, enctext dim)
            feat: (torch size: batch, maxlen, feat dim)
            feat_len: (torch size: batch)
            speaker_feat: (torch size: batch,1,speaker feat dim)

        Returns:

        """

        # Text Embedding
        feat,feat_len = self.prenet(feat,feat_len)

        # mask generation for the input text
        feat_mask = make_mask_from_len(feat_len)
        if feat.is_cuda:
            feat_mask = feat_mask.cuda(feat.device)
        speaker_feat_len=torch.ones(speaker_feat.size(0))

        # Speaker embedding
        speaker_feat = self.spkproj_prenet(speaker_feat,speaker_feat_len)[0]   
        # speaker embedding for encoder output
        speaker_feat_enc = self.spkproj_act(self.spkproj_enc(speaker_feat,speaker_feat_len)[0])
        speaker_feat_enc = speaker_feat_enc.expand((-1,enc_text.size(1),-1))

        # speaker embedding for decoder input
        speaker_feat_dec = self.spkproj_act(self.spkproj_dec(speaker_feat,speaker_feat_len)[0])
        speaker_feat_dec = speaker_feat_dec.expand((-1,feat.size(1),-1))

        #merge features with projected spk embedding
        enc_text += speaker_feat_enc
        feat += speaker_feat_dec
        
        
        # Decoding
        dec_results = self.decoder(src=enc_text, src_mask=enc_text_mask,
                                   tgt=feat, tgt_mask=feat_mask)

        # integrate trf output with spk embedding then pass to postnet
        dec_post_preout = self.postnet(dec_results['output'],feat_len)
        
        #Speech feat prediction
        dec_post_feat = self.featnet(dec_post_preout,feat_len)
        
        #End of speech prediction
        dec_post_bern = torch.cat([dec_post_feat,dec_results['output']],dim=2)
        dec_post_bern = self.bernnet(dec_post_bern,feat_len)



        # initialize the decoder outputs
        dec_outputs = dict(
            pred_feat=dec_post_feat,
            pred_bern=dec_post_bern
        )
        # if the build-in decoder has the attention results
        if 'att' in dec_results.keys():
            dec_outputs.update(
                att=dec_results['att']
            )
        # if the build-in decoder has the hidden results
        if 'hidden' in dec_results.keys():
            dec_outputs.update(
                hidden=dec_results['hidden']
            )
        return dec_outputs