import math
from typing import Any, Callable, List, Tuple
from numpy._typing import _UnknownType
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def deep_copy_module(module: Callable, n: int, *args) -> List[Any]:
  """
    Deep copies of any module n times.
    
    Arguments:
        module: The module that will be copied.
        n: The number of copies.
        args: Pass the arguments that will be used for the module.

    Returns:
      The list of the copied modules.
  """
  return [copy.deepcopy(module(*args)) for _ in range(n)]


class Generator(nn.Module):
  def __init__(self, d_model: int, vocab_size: int):
    """
      The Generator Class defines the generation step.

      Attributes:
        linear: Linear projection from the model dimension value to the 
                vocabulary size
    """
    super().__init__()
    self.linear: nn.Linear = nn.Linear(d_model, vocab_size)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
      Forward pass for the Generator class.

      Arguments:
        x: The input tensor.

      Returns:
        Output tensor projected form the model dimension to the vocabulary size, 
        softmaxed along the last dimension.
    """
    return torch.log_softmax(self.linear(x), dim=-1)


class LayerNorm(nn.Module):
  def __init__(self, features: int, eps: float=1e-6):
    """
        The LayerNorm Class used for the normalization for the Transformer
        Class.

        Attributes:
          a: The Tensor Matrix filled with ones and used for mean and standard
             deviation normalization. 
          b: The Tensor Matrix filled with zeros and used for mean and standard
             deviation normalization. 
          eps: The value used to prevent dividing by the 0, if the standard
               deviations is 0.
    """
    super().__init__()
    self.a: nn.Parameter = nn.Parameter(torch.ones(features))
    self.b: nn.Parameter = nn.Parameter(torch.zeros(features))
    self.eps: float = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
      Forward pass for the LayerNorm class.

      Arguments:
        x: The input tensor.

      Returns:
        Output tensor which is normalized using mean and the standard
        deviation.
    """
    mean: torch.Tensor = x.mean(-1, keepdim=True)
    std: torch.Tensor = x.std(-1, keepdim=True)
    return self.b + self.a * (x - mean) / (std + self.eps)


class SublayerConnection(nn.Module):
  def __init__(self, size: int, dropout: float=0.1):
    """
      A Sublayer Connection class defines the residual connection.

      Attributes:
        norm: Define the LayerNorm class.
        dropout: Define the Dropout class.
    """
    super().__init__()
    self.norm: LayerNorm = LayerNorm(size)
    self.dropout: nn.Dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor, sublayer: _UnknownType) -> torch.Tensor:
    """
      Forward pass for the SublayerConnection class.

      Arguments:
        x: Input tensor.
        sublayer: Unknown Type. ???

      Returns:
        Output tensor.
    """
   # TODO: Fix this type.
    return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
  def __init__(self, size: int, attention: 'MultiHeadAttention',
               feed_forward_network: 'FeedForwardNetwork', dropout: float=0.1, 
               n: int=2):
    """
      A EncoderLayer class defines the encoding.

      Attributes:
        size: The size variable for the SublayerConnection class. 
        feed_forward_network: The FeedForwardNetwork class.
        dropout: The dropout chance for the Dropout class.
    """
    super().__init__()
    self.size: int = size
    self.attention: MultiHeadAttention = attention
    self.feed_forward_network: FeedForwardNetwork = feed_forward_network
    self.sublayer: nn.ModuleList = nn.ModuleList(deep_copy_module(SublayerConnection, n, size, dropout))
    
  def forward(self, x: torch.Tensor, mask: _UnknownType) -> torch.Tensor:
    """
      Forward pass for the EncoderLayer class.

      Arguments:
        x: Input tensor.
        mask: Mask for the attention layer.

      Returns:
        Output tensor.
    """
    # TODO: mask is unknown type.
    x = self.sublayer[0](x, lambda x: self.attention(x, x, x, mask))
    return self.sublayer[1](x, self.feed_forward_network)

class Encoder(nn.Module):
  def __init__(self, layer, n):
    """
      A Encoder class defines the main class for the encoding use.

      Attributes:
        layers: The encoder layers.
        norm: The normalization layer.
    """
    super().__init__()
    self.layers: nn.ModuleList = nn.ModuleList(deep_copy_module(layer, n))
    self.norm: LayerNorm = LayerNorm(layer.sz)

  def forward(self, x: torch.Tensor, mask: _UnknownType) -> torch.Tensor:
    """
      Forward pass for the Encoder class.

      Arguments:
        x: Input tensor.
        mask: Mask for the Encoder Layers.

      Returns:
        Output tensor.
    """

    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)


class DecoderLayer(nn.Module):
  def __init__(self, size: int, attention: 'MultiHeadAttention', src_attention:
               'MultiHeadAttention', feed_forward_network: 'FeedForwardNetwork',
               dropout: float=0.1, n: int=3):
    """
      A Decoder class defines the main class for the decoding use.

      Attributes:
        size: Size for SublayerConnection. 
        attention: The Multihead Attention, for the DecoderLayer.
        src_attention: The source Multihead Attention, for the DecoderLayer.
        feed_forward_network: The Feed Forward Network for the Decoder Layer
        sublayers: All Sublayer connections for the DecoderLayer.


    """
    super().__init__()
    self.size: int = size
    self.attention: MultiHeadAttention = attention
    self.src_attention: MultiHeadAttention = src_attention
    self.feed_forward_network: FeedForwardNetwork = feed_forward_network
    self.sublayers: nn.ModuleList = nn.ModuleList(
      deep_copy_module(SublayerConnection, n, size, dropout)
      )

  def forward(self, x: torch.Tensor, mem: torch.Tensor, 
              src_mask: torch.Tensor, trg_mask: torch.Tensor) -> torch.Tensor:
    """
      Forward pass for the Decoder class.

      Arguments:
        x: Input tensor.
        mem: 
        src_mask: Mask for the Encoder Layers.
        trg_mask

      Returns:
        Output tensor.
    """

    x = self.sublayers[0](x, lambda x: self.attention(x, x, x, trg_mask))
    x = self.sublayers[1](x, lambda x: self.src_attention(x, mem, mem, src_mask))
    return self.sublayers[2](x, self.feed_forward_network)


class NLPDecoder(nn.Module):
  def __init__(self, layer: DecoderLayer, n: int):
    """
      NLP Decoder class.

      Attributes:
        layers: The List of Decoder Layers.
        norm: Normalization layer.
    """
    super().__init__()
    self.use_cuda = torch.cuda.is_available()
    self.layers: List = deep_copy_module(layer, n)
    self.norm = LayerNorm(layer.sz)

  def forward(self, x, mem, src_mask, trg_mask):
    """
      Forward pass for the Decoder class.

      Arguments:
        x: Input tensor.
        mem: 
        src_mask: Mask for the Encoder Layers.
        trg_mask

      Returns:
        Output tensor.
    """


    for layer in self.layers:
      x = layer(x, mem, src_mask, trg_mask)
    return self.norm(x)


class Transformer(nn.Module):
  def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
    """
      Transformer class.

      Attributes:
        encoder: The NLP Encoder.
        decoder: The NLP Decoder.



    """
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.trg_embed = trg_embed
    self.generator = generator

  def encode(self, src, src_mask):
    return self.encoder(self.src_embed(src), src_mask)

  def decode(self, mem, src_mask, trg, trg_mask):
    return self.decoder(self.trg_embed(trg), mem, src_mask, trg_mask)

  def forward(self, src, trg, src_mask, trg_mask):
    return self.decode(self.encode(src, src_mask), src_mask, trg, trg_mask)


def sub_mask(sz):
  attn_shape = (1, sz, sz)
  mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
  return mask == 0


def get_attn(q, k, v, mask=None, dropout=None) -> Tuple[torch.Tensor, torch.Tensor]:
  """
    Compute the Scaled Dot Product Attention.

    Arguments:
      q: Embedded queries.
      k: Embedded keys.
      v: Embedded values.
      mask: Mask if defined, otherwise function defined.
      dropout: Dropout chance if defined, otherwise function defined.

    Returns:
      Matrix of values masked.
      Softmaxed attention with mask.
  """
  dim_k = q.shape[-1]
  k_t = k.transpose(-2, -1)
  qk_t = torch.matmul(q, k_t)
  attn = qk_t / math.sqrt(dim_k)
  if mask is None:
    attn = attn.masked_fill(sub_mask(attn.shape[-1]), -1e9)
  sattn = attn.softmax(dim=-1)
  if dropout is not None:
    sattn = dropout(sattn)
  return torch.matmul(sattn, v), sattn


class MultiHeadAttention(nn.Module):
  def __init__(self, h, d_model, dropout=0.1):
    super().__init__()
    assert d_model == h * 64, "[ERROR] Model dimension doesn't match your"
    "parallel attention layers"
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
    del q, k, v
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
    x = x + self.pe[:, : x.size(1)]
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
