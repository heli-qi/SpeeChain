import warnings
import torch

from typing import Dict

from speechain.module.abs import Module
from speechain.module.prenet.embed import EmbedPrenet
from speechain.module.transformer.encoder import TransformerEncoder
from speechain.module.postnet.token import TokenPostnet

from speechain.utilbox.train_util import make_mask_from_len


class LanguageModel(Module):
    """
    Stand-Alone module of the autoregressive language model. This module has two usages:
        1. language model training by speechain.model.lm.LM
        2. ASR-LM joint decoding by speechain.model.ar_asr.ARASR

    """
    embedding_class_dict = dict(
        embed=EmbedPrenet
    )

    encoder_class_dict = dict(
        transformer=TransformerEncoder
    )

    def module_init(self, vocab_size: int, emb: Dict, encoder: Dict):
        """

        Args:
            vocab_size:
            emb:
            encoder:

        """
        # LM embedding layer
        assert 'type' in emb.keys(), "There must a key named 'type' in model['module_conf']['embedding']!"
        embedding_class = self.embedding_class_dict[emb['type']]
        emb['conf'] = dict() if 'conf' not in emb.keys() else emb['conf']
        self.embedding = embedding_class(vocab_size=vocab_size, **emb['conf'])

        # LM encoder part
        assert 'type' in encoder.keys(), "There must a key named 'type' in model['module_conf']['encoder']!"
        encoder_class = self.encoder_class_dict[encoder['type']]
        encoder['conf'] = dict() if 'conf' not in encoder.keys() else encoder['conf']
        # the LM encoder is automatically set to unidirectional
        encoder['conf']['uni_direction'] = True
        self.encoder = encoder_class(input_size=self.embedding.output_size, **encoder['conf'])

        # LM token prediction layer
        self.postnet = TokenPostnet(input_size=self.encoder.output_size, vocab_size=vocab_size)

    def forward(self, text: torch.Tensor, text_len: torch.Tensor):
        """

        Args:
            text:
            text_len:

        Returns:

        """
        # Text Embedding
        emb_text = self.embedding(text)

        # mask generation for the input text
        text_mask = make_mask_from_len(text_len)
        if text.is_cuda:
            text_mask = text_mask.cuda(text.device)

        # Encoding
        enc_returns = self.encoder(src=emb_text, mask=text_mask)
        # Transformer-based encoder additionally returns the encoder self-attention
        if len(enc_returns) == 4:
            enc_feat, enc_feat_mask, enc_attmat, enc_hidden = enc_returns
        # RNN-based encoder doesn't return any attention
        elif len(enc_returns) == 3:
            (enc_feat, enc_feat_mask, enc_hidden), enc_attmat = enc_returns, None
        else:
            raise RuntimeError

        # Token prediction
        logits = self.postnet(enc_feat)

        return logits, enc_feat_mask, enc_attmat
