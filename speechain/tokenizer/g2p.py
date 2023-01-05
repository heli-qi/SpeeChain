import torch
from g2p_en import G2p

from speechain.tokenizer.abs import Tokenizer


class GraphemeToPhonemeTokenizer(Tokenizer):
    """
    Tokenizer implementation that converts the input sentence string into phoneme tokens by the g2p package.

    References: https://github.com/Kyubyong/g2p

    """
    def text2tensor(self, text: str, no_sos: bool = False, no_eos: bool = False) -> torch.LongTensor:
        """
        This text-to-tensor function can take two types of input:
        1. raw string of the transcript sentence
        2. structured string of the phonemes dumped in advance

        But we recommend you to feed the type no.2 to this function because if the input it type no.1, the raw string
        needs to be decoded by g2p_en.G2p in each epoch, which not only consumes a lot of CPU but also slow down the
        model forward.

        Args:
            text: str
            no_sos:
            no_eos:

        Returns: torch.LongTensor

        """
        # initialize the tensor as an empty list
        tokens = []
        # whether to attach sos at the beginning of the tokens
        if not no_sos:
            tokens.append(self.sos_eos_idx)

        # when input text is a dumped phoneme list
        if text.startswith('[') and text.endswith(']'):
            text = text[1:-1]
            # split the text into individual tokens by a comma followed a blank
            text = text.split(', ')
            # remove the single quote marks surrounding each token if needed
            text = [token[1:-1] if token.startswith('\'') and token.endswith('\'') else token for token in text]
            tokens += [self.token2idx[token] if token in self.token2idx.keys() else self.unk_idx for token in text]
        # when input text is a raw string
        else:
            # initialize g2p convertor lazily during training
            if not hasattr(self, 'g2p'):
                self.g2p = G2p()
            phonemes = self.g2p(text)
            tokens += [self.token2idx[phn] if phn in self.token2idx.keys() else self.unk_idx for phn in phonemes]

        # whether to attach eos at the end of the tokens
        if not no_eos:
            tokens.append(self.sos_eos_idx)
        return torch.LongTensor(tokens)
