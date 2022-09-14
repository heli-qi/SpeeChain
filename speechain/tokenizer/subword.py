"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from speechain.tokenizer.abs import Tokenizer
import sentencepiece as spm
import torch

class SubwordTokenizer(Tokenizer):
    """

    """
    def tokenizer_init(self, token_model: str):
        """
        Initialize the SentencePiece tokenizer model.
        The actual subword tokenization is done by SP model instead of the vocab Dict for the speed.

        Args:
            token_model: str
                The path of the target tokenizer model file.

        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(token_model)


    def tensor2text(self, tensor: torch.Tensor):
        text = self.sp.decode_ids(tensor.tolist())
        return text


    def text2tensor(self, text: str):
        tensor = torch.LongTensor([self.sos_eos_idx] + self.sp.encode_as_ids(text) + [self.sos_eos_idx])
        return tensor
