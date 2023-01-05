import torch

from speechain.criterion.abs import Criterion


class AttentionGuidance(Criterion):
    """
    This criterion is the attention guidance loss function.

    References: Efficiently trainable text-to-speech system based on deep convolutional networks with guided attention
        https://arxiv.org/pdf/1710.08969

    """
    def criterion_init(self, sigma: float = 0.2):
        """

        Args:
            sigma:

        Returns:

        """
        self.coeff = -1 / (2 * sigma ** 2)

    def get_weight_matrix(self, X: int, Y: int) -> torch.Tensor:
        """

        Args:
            X:
            Y:

        Returns:

        """
        grid_x, grid_y = torch.meshgrid(torch.arange(X), torch.arange(Y))
        return 1 - torch.exp(self.coeff * torch.pow(grid_x / X - grid_y / Y, 2))

    def __call__(self, att_tensor: torch.Tensor, x_len: torch.Tensor, y_len: torch.Tensor = None):
        """

        Args:
            att_tensor: (batch, layer_num * head_num, max_xlen, max_ylen)
            x_len: (batch,)
            y_len: (batch,) = None

        Returns:

        """
        # argument checking
        assert len(att_tensor) == len(x_len)
        if y_len is None:
            y_len = x_len
        else:
            assert len(att_tensor) == len(y_len)

        # solve length mismatch
        if x_len.max() > att_tensor.size(2):
            x_len = x_len - (x_len.max() - att_tensor.size(2))
        if y_len.max() > att_tensor.size(3):
            y_len = y_len - (y_len.max() - att_tensor.size(3))

        # guidance weight matrix initialization, (batch, 1, max_xlen, max_ylen)
        weight_tensor = torch.zeros((att_tensor.size(0), 1, att_tensor.size(2), att_tensor.size(3)),
                                    device=att_tensor.device)
        # guidance mask matrix initialization, (batch, 1, max_xlen, max_ylen)
        mask_flag = torch.zeros(weight_tensor.size(), dtype=torch.bool, device=weight_tensor.device)
        # loop each utterance and register its weight and mask matrices
        for i, (X, Y) in enumerate(zip(x_len, y_len)):
            X, Y = X.item(), Y.item()
            weight_tensor[i][0][:X, :Y] = self.get_weight_matrix(X, Y).to(att_tensor.device)
            mask_flag[i][0][:X, :Y] = 1

        # return the mean value of the masked results
        return torch.mean(torch.masked_select(att_tensor * weight_tensor, mask_flag))
