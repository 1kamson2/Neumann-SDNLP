import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


class NLPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.mha = MultiHeadAttention(7, 7, 7, 8)
        self.ffn = FeedForwardNetwork(7, 14)

    def forward(self, ex, q, k, v, valid_lens):
        out = ex + self.mha(q, k, v, valid_lens)
        out = nn.LayerNorm(out)
        out = out + self.ffn(out)
        return out


class NLPDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.mha = MultiHeadAttention(8, 8, 8, 8)
        self.ffn = FeedForwardNetwork(8, 16)

    def forward(self, dx, ex):
        out = dx + self.mha(dx)
        out = nn.LayerNorm(out)
        out = out + self.mha(out + ex)
        out = nn.LayerNorm(out)
        out = out + self.ffn(out)
        out = nn.LayerNorm(out)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.encoder = NLPEncoder()
        self.decoder = NLPDecoder()
        # skip for now
        self.lin = nn.Linear(10000, 100000)

    def forward(self, dx, ex):
        ex = self.encoder(ex)
        out = self.decoder(dx, ex)
        out = self.lin(out)
        return nn.Softmax(out)


"""
100% WON'T WORK THIS SHIT BELOW, FUCKING PIECE OF SHIT
INPUT SHOULD BE THE LENGTH OF THE EMBEDDING IN THIS CASE IT IS 7 OR 8
IF IT WAS ONE HOT ENCODING IT WOULD BE 128 OR 256
https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
"""


class MultiHeadAttention(nn.Module):
    def __init__(self, k_sz, q_sz, v_sz, num_heads, dropout=0.1, bias=False):
        super().__init__()
        self.num_heads = num_heads
        # Weights of each layer
        self.WL_k = nn.Linear(k_sz, k_sz * num_heads, bias=bias)
        self.WL_q = nn.Linear(q_sz, q_sz * num_heads, bias=bias)
        self.WL_v = nn.Linear(v_sz, v_sz * num_heads, bias=bias)
        self.dropout = dropout
        # Output layer - fully connected layer
        # should use d_model = 7 or 8 for better clarity
        self.fc = nn.Linear(v_sz * num_heads, v_sz, bias=bias)

    def forward(self, q, k, v, valid_lens):
        q = self._transpose(self.WL_q(q), self.num_heads)
        k = self._transpose(self.WL_q(k), self.num_heads)
        v = self._transpose(self.WL_q(v), self.num_heads)
        valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        # For now use this, but later do your own version
        output = F.scaled_dot_product_attention(q, k, v, valid_lens, self.dropout)
        output_concat = self._transpose_output(output, self.num_heads)
        return self.fc(output_concat).to(torch.device("cpu"))

    def _transpose(self, X, num_heads):
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def _transpose_output(self, X, num_heads):
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)


class FeedForwardNetwork(nn.Module):
    def __init__(self, dim_in, dim_hidout, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_hidout)
        self.w2 = nn.Linear(dim_hidout, dim_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        return x


"""
TODO: Add masked attention, for now skipped

"""
