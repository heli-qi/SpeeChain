import torch

from typing import List
from g2p_en import G2p

from speechain.tokenizer.abs import Tokenizer


# some abnormal phonemes G2P may give during decoding
abnormal_phns = ['...', '. ...', '... .',
                 '. .','. . .', '. . . .', '. . . . . . .', '. . . . . .', '. . . . . . . .', '. . . . .', '. . . . . . . . .',
                 '..', '.. ..']
cmu_phn_list = ['AH', 'AH0', 'AH1', 'AH2', 'IH', 'IH0', 'IH1', 'IH2', 'EH', 'EH0', 'EH1', 'EH2', 'AE', 'AE0', 'AE1', 'AE2',
                'ER', 'ER0', 'ER1', 'ER2', 'UW', 'UW0', 'UW1', 'UW2', 'IY', 'IY0', 'IY1', 'IY2', 'AA', 'AA0', 'AA1', 'AA2',
                'AY', 'AY0', 'AY1', 'AY2', 'AO', 'AO0', 'AO1', 'AO2', 'EY', 'EY0', 'EY1', 'EY2', 'OW', 'OW0', 'OW1', 'OW2',
                'AW', 'AW0', 'AW1', 'AW2', 'UH', 'UH0', 'UH1', 'UH2', 'OY', 'OY0', 'OY1', 'OY2',
                'T', 'N', 'D', 'S', 'R', 'L', 'DH', 'M', 'K', 'Z', 'W', 'HH', 'P', 'V', 'F', 'B', 'NG', 'G', 'SH', 'Y', 'CH', 'TH', 'JH', 'ZH']

class GraphemeToPhonemeTokenizer(Tokenizer):
    """
    Tokenizer implementation that converts the input sentence string into phoneme tokens by the g2p package.

    References: https://github.com/Kyubyong/g2p

    """
    def text2tensor(self, text: str or List[str], no_sos: bool = False, no_eos: bool = False, return_tensor: bool = True):
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

        if return_tensor:
            return torch.LongTensor(tokens)
        else:
            return tokens
