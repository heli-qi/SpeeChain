import torch

from speechain.criterion.abs import Criterion
from speechain.utilbox.train_util import make_mask_from_len

class Perplexity(Criterion):
    """

    """
    def __call__(self, logits: torch.Tensor, text: torch.Tensor, text_len: torch.Tensor):
        """

        Args:
            logits:
            text:
            text_len:

        Returns:

        """
        # mask generation for the input text
        text_mask = make_mask_from_len(text_len - 1, return_3d=False)
        if text.is_cuda:
            text_mask = text_mask.cuda(text.device)

        # perplexity calculation
        log_prob = torch.log_softmax(logits, dim=-1)
        text_prob = log_prob.gather(-1, text[:, 1:].view(text.size(0), -1, 1)).squeeze(dim=-1)
        text_prob = text_prob.masked_fill(~text_mask, 0.0)
        text_ppl = torch.exp(torch.sum(text_prob, dim=-1) * (- 1 / (text_len - 1))).mean()

        return text_ppl
