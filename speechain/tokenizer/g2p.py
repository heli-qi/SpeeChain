import torch

from typing import List
from g2p_en import G2p

from speechain.tokenizer.abs import Tokenizer


# some abnormal phonemes G2P may give during decoding
abnormal_phns = ['...', '. . .', '. . . .', '..', '. ...', '. .', '. . . . . . .', '. . . . . .',
                 '. . . . . . . .', '. . . . .', '.. ..', '... .', '. . . . . . . . .']

class GraphemeToPhonemeTokenizer(Tokenizer):
    """
    Tokenizer implementation that converts the input sentence string into phoneme tokens by the g2p package.

    References: https://github.com/Kyubyong/g2p

    """
    def text2tensor(self, text: str or List[str], no_sos: bool = False, no_eos: bool = False) -> torch.LongTensor:
        """
        This text-to-tensor function can take two types of input:
        1. raw string of the transcript sentence
        2. structured string of the phonemes dumped in advance

        But we recommend you to feed the type no.2 to this function because if the input it type no.1, the raw string
        needs to be decoded by g2p_en.G2p in each epoch, which not only consumes a lot of CPU but also slow down the
        model forward.

        """
        # initialize the tensor as an empty list
        tokens = []
        # whether to attach sos at the beginning of the tokens
        if not no_sos:
            tokens.append(self.sos_eos_idx)

        # when input text is a dumped phoneme list
        if isinstance(text, List):
            tokens += [self.token2idx[token] if token in self.token2idx.keys() else self.unk_idx for token in text]
        # when input text is a raw string
        else:
            # initialize g2p convertor lazily during training
            if not hasattr(self, 'g2p'):
                self.g2p = G2p()
            phonemes = self.g2p(text)
            for phn in phonemes:
                if phn in abnormal_phns:
                    continue
                elif phn == ' ':
                    tokens.append(self.space_idx)
                elif phn not in self.token2idx.keys():
                    tokens.append(self.unk_idx)
                else:
                    tokens.append(self.token2idx[phn])

        # whether to attach eos at the end of the tokens
        if not no_eos:
            tokens.append(self.sos_eos_idx)
        return torch.LongTensor(tokens)
