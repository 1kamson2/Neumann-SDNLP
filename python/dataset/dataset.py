from typing import List
from datasets import Dataset, load_dataset
from spacy.language import Language
import torch
import logging
from enum import Enum
from torch._prims_common import DeviceLikeType
from torchtext.vocab import Vocab
from models.nlp.model import sub_mask
import torch.nn as nn
from torch.nn.functional import pad
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.tokenization import tokenize

"""
    THIS SHOULD BE ONLY USED ON WINDOWS (IS THAT EVEN TRUE?) 
    from os.path import join
"""
logger = logging.getLogger(__name__)


class DataType(Enum):
  TRAINING = 0
  VALIDATION = 1
  TEST = 2


class LabelSmoothing(nn.Module):
  """
  Class that implements regularization for your data.
  """

  def __init__(self, size, padding_idx, smoothing=0.0):
    super(LabelSmoothing, self).__init__()
    self.kldivloss = nn.KLDivLoss(reduction="sum")
    self.padding_idx = padding_idx
    self.confidence = 1.0 - smoothing
    self.smoothing = smoothing
    self.size = size
    self.true_dist = None

  def forward(self, x, target):
    assert x.size(1) == self.size
    true_dist = x.data.clone()
    true_dist.fill_(self.smoothing / (self.size - 2))
    true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
    true_dist[:, self.padding_idx] = 0
    mask = torch.nonzero(target.data == self.padding_idx)
    if mask.dim() > 0:
      true_dist.index_fill_(0, mask.squeeze(), 0.0)
    self.true_dist = true_dist
    return self.kldivloss(x, true_dist.clone().detach())


class Loss:
  def __init__(self, generator, criterion):
    self.generator = generator
    self.criterion = criterion

  def __call__(self, x, y, norm):
    x = self.generator(x)
    loss = (
      self.criterion(
        x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
      )
      / norm
    )
    return loss.data * norm, loss


class Batch:
  def __init__(
    self, src, tgt, pad=2, finit=False
  ):  
    # Default padding is <blank>
    if finit is False:
      self.src = src
      self.src_mask = (src != pad).unsqueeze(-2)
      if tgt is not None:
        self.tgt = tgt[:, :-1]
        self.tgt_y = tgt[:, 1:]
        self.tgt_mask = self.make_std_mask(self.tgt, pad)
        self.n_toks = (self.tgt_y != pad).data.sum()

  @staticmethod
  def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & sub_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask

  def collate_batch(
    self,
    batch,
    nlp_src: Language,
    nlp_tgt: Language,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    device: DeviceLikeType,
    padding: int = 128,
    pad_id: int = 2,
  ):
    bs_id = torch.tensor([0], device=device)  # <s> token ID
    eos_id = torch.tensor([1], device=device)  # </s> token ID
    src_toks, tgt_toks = [], []

    for src_tok, tgt_tok in batch:
      processed_src = torch.cat(
        [
          bs_id,
          torch.tensor(
            vocab_src(nlp_src(src_tok)), dtype=torch.int64, device=device
          ),
          eos_id,
        ],
        0,
      )
      processed_tgt = torch.cat(
        [
          bs_id,
          torch.tensor(
            vocab_tgt(nlp_tgt(tgt_tok)), dtype=torch.int64, device=device
          ),
          eos_id,
        ],
        0,
      )
      src_toks.append(
        pad(processed_src, (0, padding - len(processed_src)), value=pad_id)
      )
      tgt_toks.append(
        pad(processed_tgt, (0, padding - len(processed_tgt)), value=pad_id)
      )

    return torch.stack(src_toks), torch.stack(tgt_toks)

  def get_dataloader(
    self,
    device: DeviceLikeType,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    nlp_de: Language,
    nlp_en: Language,
    batch_sz: int = 12000,
    padding: int = 128,
    is_distributed: bool = False,
  ):
    def __de_pipeline(text: str) -> List:
      """
        Wrapper for the decode vocabulary tokenization, pipeline. 

        Arguments:
          text: Text used in pipeline. 

        Returns:
          List of tokens. 
      """
      return tokenize(text, nlp_de)

    def __en_pipeline(text: str) -> List:
      """
        Wrapper for the encode vocabulary tokenization, pipeline. 

        Arguments:
          text: Text used in pipeline. 

        Returns:
          List of tokens. 
      """
      return tokenize(text, nlp_en)

    def __collate_fn(batch):
      return self.collate_batch(
        batch,
        __de_pipeline,
        __en_pipeline,
        vocab_src,
        vocab_tgt,
        device,
        padding=padding,
        pad_id=2,
      )

    train_iter = load_dataset("bentrevett/multi30k", split="train") 
    valid_iter = load_dataset("bentrevett/multi30k", split="validation") 
    assert isinstance(train_iter, Dataset)
    assert isinstance(valid_iter, Dataset)
    train_iter_map = to_map_style_dataset(train_iter)
    valid_iter_map = to_map_style_dataset(valid_iter)
    train_sampler = (
      DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_sampler = (
      DistributedSampler(valid_iter_map) if is_distributed else None
    )
    train_dl = DataLoader(
      train_iter_map,
      batch_size=batch_sz,
      shuffle=(train_sampler is None),
      sampler=train_sampler,
      collate_fn=__collate_fn,
    )
    valid_dl = DataLoader(
      valid_iter_map,
      batch_size=batch_sz,
      shuffle=(valid_sampler is None),
      sampler=valid_sampler,
      collate_fn=__collate_fn,
    )
    return train_dl, valid_dl
