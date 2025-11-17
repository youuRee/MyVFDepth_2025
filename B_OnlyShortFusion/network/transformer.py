import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformer(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        self.linear = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(embed_dims, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dims)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, q, k, v, src_mask=None, src_key_padding_mask=None):
        attn_output, attn_weights = self.cross_attn(q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        attended_values = v + self.dropout1(attn_output)
        out = self.norm1(attended_values)
        '''
        src2 = self.linear2(self.dropout(F.relu(self.linear1(attended_values))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        '''
        return out