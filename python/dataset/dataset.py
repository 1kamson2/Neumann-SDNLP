import torch
import logging
from enum import Enum
from models.nlp.model import sub_mask
import torch.nn as nn
from torch.nn.functional import pad
import torchtext.datasets as datasets
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

"""
    THIS SHOULD BE ONLY USED ON WINDOWS (IS THAT EVEN TRUE?) 
    from os.path import join
"""
logger = logging.getLogger(__name__)


class DataType(Enum):
  TRAINING = 0
  VALIDATION = 1
  TEST = 2


class Language(Enum):
  SRC = "source"
  TRG = "target"


class FileManager:
  root_project_path: Path = Path(
    "/home/kums0n-desktop/Dev/lm-project/neumann-lm/"
  )
  python_catalog: Path = root_project_path.joinpath("python/")
  python_data_catalog: Path = python_catalog.joinpath("data/spacy_vocab.pt")
  gpt_out_path: Path = python_catalog.joinpath("GPT_OUT")
  gpt_api_path: Path = python_catalog.joinpath("API_KEY")
  unet_weights_path: Path = python_catalog.joinpath("weights/UNET_WEIGHTS.pth")
  ddpm_weights_path: Path = python_catalog.joinpath("weights/DDPM_WEIGHTS.pth")
  transformer_weights_path: Path = python_catalog.joinpath(
    "weights/TRANSFORMER_WEIGHTS.pth"
  )


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
  def make_std_mask(trg, pad):
    "Create a mask to hide padding and future words."
    trg_mask = (trg != pad).unsqueeze(-2)
    trg_mask = trg_mask & sub_mask(trg.size(-1)).type_as(trg_mask.data)
    return trg_mask

  def collate_batch(
    self,
    batch,
    src_pipeline,
    trg_pipeline,
    src_vocab,
    trg_vocab,
    device,
    padding=128,
    pad_id=2,
  ):
    bs_id = torch.tensor([0], device=device)  # <s> token ID
    eos_id = torch.tensor([1], device=device)  # </s> token ID
    src_list, trg_list = [], []

    for _src, _trg in batch:
      processed_src = torch.cat(
        [
          bs_id,
          torch.tensor(
            src_vocab(src_pipeline(_src)), dtype=torch.int64, device=device
          ),
          eos_id,
        ],
        0,
      )
      processed_trg = torch.cat(
        [
          bs_id,
          torch.tensor(
            trg_vocab(trg_pipeline(_trg)), dtype=torch.int64, device=device
          ),
          eos_id,
        ],
        0,
      )
      src_list.append(
        pad(processed_src, (0, padding - len(processed_src)), value=pad_id)
      )
      trg_list.append(
        pad(processed_trg, (0, padding - len(processed_trg)), value=pad_id)
      )

    src = torch.stack(src_list)
    trg = torch.stack(trg_list)
    return src, trg

  def get_dataloader(
    self,
    device,
    vocab_src,
    vocab_trg,
    vocab2dec,
    vocab2enc,
    batch_sz=12000,
    padding=128,
    is_distributed=True,
  ):
    # Tokenizer functions
    def tokenize_de(text):
      return Tokenizer.tokenize(text, vocab2dec)

    def tokenize_en(text):
      return Tokenizer.tokenize(text, vocab2enc)

    def collate_fn(batch):
      return self.collate_batch(
        batch,
        tokenize_de,
        tokenize_en,
        vocab_src,
        vocab_trg,
        device,
        padding=padding,
        pad_id=2,
      )

    train_iter = datasets.Multi30k(split="train", language_pair=("de", "en"))
    valid_iter = datasets.Multi30k(split="valid", language_pair=("de", "en"))

    train_iter_map = to_map_style_dataset(train_iter)
    train_sampler = (
      DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
      DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dl = DataLoader(
      train_iter_map,
      batch_size=batch_sz,
      shuffle=(train_sampler is None),
      sampler=train_sampler,
      collate_fn=collate_fn,
    )
    valid_dl = DataLoader(
      valid_iter_map,
      batch_size=batch_sz,
      shuffle=(valid_sampler is None),
      sampler=valid_sampler,
      collate_fn=collate_fn,
    )
    return train_dl, valid_dl



"""
KNOWN ISSUES:
> https://github.com/pytorch/text/issues/2221
  Might be issue with downloading test data.
"""
