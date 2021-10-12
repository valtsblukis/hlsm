import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, n_head, dim_ff, dropout=0.1, kvdim=None):
        super().__init__()
        kvdim = kvdim if kvdim is not None else d_model
        self.mh_attention = nn.MultiheadAttention(d_model, n_head, dropout, kdim=kvdim, vdim=kvdim)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = nn.LeakyReLU()

    def forward(self, src_labels: torch.Tensor, attn_mask: torch.Tensor, inputs_are_labels=True, return_attn=False):
        src_labels = src_labels[:, None, :]
        # Create an extra "batch" dimension, and treat the current batch dimension as a pos dimension
        seq_mask = attn_mask
        x, attn_w = self.mh_attention(src_labels, src_labels, src_labels, attn_mask=attn_mask)

        # inputs contain information about ground truth of the outputs for each element
        # and thus cannot be added with a residual connection.
        # The attn_mask is responsible for preventing label leakage.
        if inputs_are_labels:
            x = self.dropout(x)
        else:
            x = src_labels + self.dropout(x)

        x = self.norm1(x)
        y = self.linear2(self.dropout(self.act(self.linear1(x))))
        y = x + self.dropout2(y)
        y = self.norm2(y)
        if return_attn:
            return y[:, 0, :], attn_w
        else:
            return y[:, 0, :]
