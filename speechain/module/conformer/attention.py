import torch

from torch import nn
from speechain.module.transformer.attention import MultiHeadedAttention

class RelPosMultiHeadedAttention(MultiHeadedAttention):
    """

    """

    def module_init(self, num_heads: int, d_model: int, dropout: float = 0.1, scale_dp_by_head: bool = False):
        super(RelPosMultiHeadedAttention, self).module_init(num_heads, d_model, dropout, scale_dp_by_head)

        self.pos_layer = nn.Linear(d_model, d_model, bias=False)

        self.pos_bias_u = nn.Parameter(torch.Tensor(self.num_heads, self.head_size))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.num_heads, self.head_size))

    def rel_shift(self, matrix_bd: torch.Tensor):

        # (batch_size, num_heads, seq_len, 1)
        zero_pad = torch.zeros((*matrix_bd.size()[:3], 1), device=matrix_bd.device, dtype=matrix_bd.dtype)
        # (batch_size, num_heads, seq_len, 2 * seq_len)
        matrix_bd_padded = torch.cat([zero_pad, matrix_bd], dim=-1)

        # (batch_size, num_heads, 2 * seq_len, seq_len)
        matrix_bd_padded = matrix_bd_padded.view(*matrix_bd_padded.size()[:2], matrix_bd_padded.size(3), matrix_bd_padded.size(2))
        # (batch_size, num_heads, seq_len)
        matrix_bd = matrix_bd_padded[:, :, 1:].view_as(matrix_bd)[
            :, :, :, : matrix_bd.size(-1) // 2 + 1
            ]  # only keep the positions from 0 to time2

        return matrix_bd

    def forward(self, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor, mask: torch.Tensor = None, posenc: torch.Tensor = None):

        assert posenc is not None, "posnec must be given for RelPosMultiHeadedAttention!"

        # (batch_size, num_heads, seq_len_q or seq_len_kv, head_size)
        k, v, q = self.kvq_forward(k, v, q)

        # (batch_size, num_heads, seq_len_q, head_size)
        q_with_bias_u = q + self.pos_bias_u[None, :, None, :]
        q_with_bias_v = q + self.pos_bias_v[None, :, None, :]

        # (batch_size, num_heads, 2 * seq_len - 1, head_size)
        posenc = self.pos_layer(posenc).view(posenc.size(0), -1, self.num_heads, self.head_size).transpose(1, 2)

        # (batch_size, num_heads, seq_len_q, head_size) * (batch_size, num_heads, head_size, seq_len_kv) = (batch_size, num_heads, seq_len_q, seq_len_kv)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(2, 3))

        # (batch_size, num_heads, seq_len_q, head_size) * (batch_size, num_heads, head_size, 2 * seq_len_q - 1) = (batch_size, num_heads, seq_len, 2 * seq_len_q - 1)
        matrix_bd = torch.matmul(q_with_bias_v, posenc.transpose(2, 3))
        # (batch_size, num_heads, seq_len_q, seq_len_q)
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) * self.scale
        return self.attention_forward(v, scores, mask)
