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
    def tokenizer_init(self, token_model: str,
                       enable_sampling: bool = False, alpha: float = 0.0, nbest_size: int = -1):
        """
        Initialize the SentencePiece tokenizer model.
        The actual subword tokenization is done by SP model instead of the vocab Dict for the speed.

        Args:
            token_model: str
                The path of the target tokenizer model file.
            enable_sampling: bool
                Whether to enable encode sampling for the tokenizer. Only effective to unigram tokenizer.
            alpha: float
                Inverse temperature for encoding sampling. Only effective to unigram tokenizer.
            nbest_size: int
                The number of tokenization results returned for each sentence. Only effective to unigram tokenizer.

        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(token_model)

        self.encode_conf = dict(
            enable_sampling=enable_sampling,
            alpha=alpha, nbest_size=nbest_size
        )


    def tensor2text(self, tensor: torch.Tensor):
        text = self.sp.decode_ids(tensor.tolist())
        return text


    def text2tensor(self, text: str):
        tensor = torch.LongTensor([self.sos_eos_idx] +
                                  self.sp.encode_as_ids(text, **self.encode_conf) + [self.sos_eos_idx])
        return tensor
