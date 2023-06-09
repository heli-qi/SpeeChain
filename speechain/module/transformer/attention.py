"""
    Origin: Sashi Novitasari
    Modification: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import math
import torch
import torch.nn as nn

from speechain.module.abs import Module


class MultiHeadedAttention(Module):
    """
    A Multi-Head Attention layer has:
        · Query linear layer
        · Key linear layer
        · Value linear layer
        · Softmax layer
        · Attention Dropout layer
        · Output linear layer

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def module_init(self, num_heads: int, d_model: int, dropout: float = 0.1, scale_dp_by_head: bool = False):
        """
        Create a multi-headed attention layer.

        Args:
            num_heads:
                The number of heads
            d_model:
                Model size (must be divisible by num_heads)
            dropout:
                The dropout rate of the Dropout layer after the softmax operation
        """
        assert d_model % num_heads == 0, "d_model is not divisible by num_heads!"

        self.head_size = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads

        self.k_layer = nn.Linear(d_model, num_heads * self.head_size)
        self.v_layer = nn.Linear(d_model, num_heads * self.head_size)
        self.q_layer = nn.Linear(d_model, num_heads * self.head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, d_model)

        self.scale = 1 / math.sqrt(self.head_size) if scale_dp_by_head else 1 / math.sqrt(self.d_model)

    def kvq_forward(self, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor):

        batch_size = k.size(0)

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # separate all heads of q, k, v
        k = k.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        return k, v, q

    def attention_forward(self, v: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor):

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        score_soft = attention.clone()
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(
            v.size(0), -1, self.num_heads * self.head_size
        )

        output = self.output_layer(context)

        return output, score_soft

    def forward(self, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor, mask: torch.Tensor = None):
        """
        Computes multi-headed attention.

        Args:
            k: keys   [B, M, D] with M being the sentence length.
            v: values [B, M, D]
            q: query  [B, M, D]
            mask: optional mask [B, 1, M]

        Returns:

        """

        k, v, q = self.kvq_forward(k, v, q)

        # compute scaled attention scores
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scale

        return self.attention_forward(v, scores, mask)
