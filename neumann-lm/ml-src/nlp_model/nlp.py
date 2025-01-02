import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1) 


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        """
        Create Layer normalization for Transformer. Using ones for a and zeros
        for b because they are at first neutral. Using nn.Parameter(...) to make
        it trainable.
        """
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.b + self.a * (x - mean) / (std + self.eps)

class SublayerCon(nn.Module):
    def __init__(self, sz, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(sz)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) 


class EncoderLayer(nn.Module):
    def __init__(self, sz, attn, ffn, dropout, n=2):
        super().__init__()
        self.sz = sz
        self.attn = attn
        self.ffn = ffn
        self.sublayer = nn.ModuleList(
            [copy.deepcopy(SublayerCon(sz, dropout)) for _ in range(n)]
        )

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer[0](x, self.ffn) 


class NLPEncoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        """
        Main Encoder class, holds layers.
        """
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.sz)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, sz, attn, src_attn, ffn, dropout=0.1, n=3):
        super().__init__()
        self.sz = sz
        self.attn = attn
        self.src_attn = src_attn
        self.ffn = ffn
        self.sublayer = nn.ModuleList(
            [copy.deepcopy(SublayerCon(sz, dropout)) for _ in range(n)]
        )

    def forward(self, x, mem, src_mask, trg_mask):
        # [TODO]: Should do m = memory?
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, trg_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, mem, mem, src_mask))
        return self.sublayer[2](x, self.ffn) 


class NLPDecoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.use_cuda = torch.cuda.is_available()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.sz)

    def forward(self, x, mem, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, mem, src_mask, trg_mask)
        return self.norm(x) 


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator

    """
    Helper functions
    """

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask) 

    def decode(self, mem, src_mask, trg, trg_mask):
        return self.decoder(self.trg_embed(trg), mem, src_mask, trg_mask) 

    def forward(self, src, trg, src_mask, trg_mask):
        return self.decode(self.encode(src, src_mask), src_mask, trg,
                           trg_mask) 


def sub_mask(sz):
    attn_shape = (1, sz, sz)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return (mask == 0) 

def get_attn(q, k, v, mask=None, dropout=None):
    """
    Compute Scaled Dot Product Attention for set of queries, keys and
    values. It accepts also mask, mainly for decoder.
    """
    dim_k = q.shape[-1]
    k_t = k.transpose(-2, -1)
    qk_t = torch.matmul(q, k_t)
    attn = qk_t / math.sqrt(dim_k)
    if mask is None:
        attn = attn.masked_fill(sub_mask(attn.shape[-1]) == 0, -1e9)
    sattn = attn.softmax(dim=-1)
    if dropout is not None:
        sattn = dropout(sattn)
    return torch.matmul(sattn, v), sattn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model == h * 64, f"[ERROR] Model dimension doesn't match your"
        f"parallel attention layers"
        self.dim_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)]
        )
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatch = q.shape[0]
        q, k, v = [
            lin(x).view(nbatch, -1, self.h, self.dim_k).transpose(1, 2)
            for lin, x in zip(self.linears, (q, k, v))
        ]
        x, self.attn = get_attn(q, k, v, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatch, -1, self.h * self.dim_k)
        # [TODO]: Should del q, k, v?
        return self.linears[-1](x) 


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x) 


class FeedForwardNetwork(nn.Module):
    def __init__(self, dim_model, dim_ffnout, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim_model, dim_ffnout)
        self.w2 = nn.Linear(dim_ffnout, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.w1(x)
        out = F.relu(out)
        out = self.dropout(out)
        return self.w2(out)
