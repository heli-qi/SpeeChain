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
    Tokenizer implementation that converts the input sentence string into a list of subword tokens
    (i.e., combinations of graphemes).

    """

    def tokenizer_init(self, token_model: str, model_package: str = 'sp'):
        """
        Initialize the subword tokenizer model.
        For subword tokenizer, tokenization is done by third-party packages instead of the built-in token Dicts.

        Args:
            token_model: str
                The path of your specified tokenizer model file.
            model_package: str
                The package you want to use to parse your tokenizer model. Default to be 'sp' (sentencepiece)

        """
        # tokenization by the sentencepiece package
        if model_package == 'sp':
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(token_model)

            # '_'-prefixed built-in function members that should not be used externally
            self._text2tensor_func = self.sp_model.encode_as_ids
            self._tensor2text_func = self.sp_model.decode_ids

        else:
            raise NotImplementedError(
                "The tokenizer packages other than sentencepiece have not been implemented yet~~~"
            )

    def tensor2text(self, tensor: torch.Tensor):
        """

        Args:
            tensor:

        Returns:

        """
        text = self._tensor2text_func(tensor.tolist())
        return text

    def text2tensor(self, text: str, no_sos: bool = False, no_eos: bool = False):
        """

        Args:
            text:
            no_sos:
            no_eos:

        Returns:

        """
        # initialize the tensor as an empty list
        tokens = []
        # whether to attach sos at the beginning of the tokens
        if not no_sos:
            tokens.append(self.sos_eos_idx)
        # attach the main body of the text
        tokens.extend(self._text2tensor_func(text))
        # whether to attach eos at the end of the tokens
        if not no_eos:
            tokens.append(self.sos_eos_idx)
        # turn the token list into a long-type tensor
        return torch.LongTensor(tokens)
