"""
    Author: Sashi Novitasari
    Affiliation: NAIST (-2022)
    Date: 2022.08

    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.09
"""
import torch
from typing import Dict
from speechain.module.abs import Module
from speechain.utilbox.train_util import make_mask_from_len

from speechain.module.prenet.embed import EmbedPrenet
from speechain.module.prenet.conv1d import Conv1dPrenet
from speechain.module.transformer.encoder import TransformerEncoder


class TTSEncoder(Module):
    """

    """
    embedding_class_dict = dict(
        emb=EmbedPrenet
    )

    prenet_class_dict = dict(
        conv1d=Conv1dPrenet
    )

    encoder_class_dict = dict(
        transformer=TransformerEncoder
    )

    def module_init(self,
                    embedding: Dict,
                    encoder: Dict,
                    prenet: Dict = None,
                    vocab_size: int = None):
        """

        Args:
            encoder:
            embedding:
            prenet:
            vocab_size:

        """
        # temporary register for connecting two sequential modules
        _prev_output_size = None

        # Token embedding layer
        embedding_class = self.embedding_class_dict[embedding['type']]
        embedding['conf'] = dict() if 'conf' not in embedding.keys() else embedding['conf']
        self.embedding = embedding_class(vocab_size=vocab_size, **embedding['conf'])
        _prev_output_size = self.embedding.output_size

        # TTS Encoder Prenet
        if prenet is not None:
            prenet_class = self.prenet_class_dict[prenet['type']]
            prenet['conf'] = dict() if 'conf' not in prenet.keys() else prenet['conf']
            self.prenet = prenet_class(input_size=_prev_output_size, **prenet['conf'])
            _prev_output_size = self.prenet.output_size

        # main body of the E2E TTS encoder
        encoder_class = self.encoder_class_dict[encoder['type']]
        encoder['conf'] = dict() if 'conf' not in encoder.keys() else encoder['conf']
        self.encoder = encoder_class(input_size=_prev_output_size, **encoder['conf'])

        # register the output size for assembly
        self.output_size = self.encoder.output_size

    def forward(self, text: torch.Tensor, text_len: torch.Tensor):
        """

        Args:
            text: (torch size: batch, maxlen)
            text_len: (torch size: batch )

        Returns:
            enc_outputs: dict
        """
        # Token Embedding
        text = self.embedding(text)

        # Prenet Processing if specified
        if hasattr(self, 'prenet'):
            text, text_len = self.prenet(text, text_len)

        # TTS Encoding
        text, text_mask, attmat, hidden = self.encoder(text, make_mask_from_len(text_len))
        return text, text_mask, attmat, hidden
