import torch

from speechain.criterion.abs import Criterion


class CTCLoss(Criterion):
    """
    The wrapper class for torch.nn.functional.ctc_loss

    """
    def criterion_init(self, blank: int = 0, zero_infinity: bool = True):
        """

        Args:
            weight: float
                The weight on the CTC loss in the overall ASR loss. Used to balance the loss terms outside this class.
            blank: int = 0
                The blank label for CTC modeling. In order to use CuDNN, blank must be set to 0.
            zero_infinity: bool = True
                Whether to zero infinite losses and the associated gradients when calculating the CTC loss.

        """
        # arguments checking
        self.blank = blank
        self.zero_infinity = zero_infinity

    def __call__(self, ctc_logits: torch.Tensor, enc_feat_len: torch.Tensor,
                 text: torch.Tensor, text_len: torch.Tensor):
        """

        Args:
            ctc_logits: (batch, enc_feat_len, vocab)
                The model output from the CTC layer before the softmax operation.
            enc_feat_len: (batch,)
                The length of encoder feature sequences (<= the length of acoustic feature sequence)
            text: (batch, text_len)
                The grount-truth token index sequences.
            text_len: (batch,)
                The length of each token index sequence.

        """
        batch, enc_feat_maxlen = ctc_logits.size(0), enc_feat_len.max().item()

        # (batch, enc_feat_len, vocab) -> (enc_feat_len, batch, vocab)
        ctc_logits = ctc_logits.transpose(0, 1).log_softmax(dim=-1)

        # remove the <sos/eos> at the beginning and end of each sentence
        text, text_len = text[:, 1:-1], text_len - 2
        if len(text.shape) == 1:
            text = text.unsqueeze(-1)
        text = torch.cat([text[i, :text_len[i]] for i in range(batch)])

        # obtain the ctc loss for each data instances in the given batch
        loss = torch.nn.functional.ctc_loss(ctc_logits, text, enc_feat_len, text_len,
                                            blank=self.blank, reduction='none', zero_infinity=self.zero_infinity)
        return loss.mean()

    def recover(self, ctc_text: torch.Tensor, ctc_text_len: torch.Tensor):

        if len(ctc_text.shape) == 3:
            ctc_text = torch.argmax(ctc_text, dim=-1)

        text = [[] for _ in range(ctc_text.size(0))]
        for i in range(ctc_text.size(0)):
            for j in range(ctc_text_len[i]):
                token = ctc_text[i][j].item()
                if (j == 0 or token != ctc_text[i][j - 1]) and token != self.blank:
                    text[i].append(token)

        return text
