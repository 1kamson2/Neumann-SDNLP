import torch
import logging
from torch.utils.data import Dataset
from enum import Enum
from functools import lru_cache
from nlp_model.nlp import sub_mask
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import torchtext.datasets as datasets
import spacy
import os
from os.path import exists
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)
NLP_TRAINING_DATA_PATH = "rsrc/WMT14_train.csv"
NLP_VALIDATION_DATA_PATH = "rsrc/WMT14_validation.csv"
NLP_TEST_DATA_PATH = "rsrc/WMT14_test.csv"
EN_WORD_EMBEDDING_LENGTH = 16
DE_WORD_EMBEDDING_LENGTH = 16


class DataType(Enum):
    TRAINING = 0
    VALIDATION = 1
    TEST = 2


class Language(Enum):
    SRC = "source"
    TRG = "target"


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
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm 
        return loss.data * norm, loss

class Batch:
    def __init__(self, src, trg, pad=2, finit=False):  # Default padding is <blank>
        if finit is False:
            self.src = src
            self.src_mask = (src != pad).unsqueeze(-2)
            if trg is not None:
                self.trg = trg[:, :-1]
                self.trg_y = trg[:, 1:]
                self.trg_mask = self.make_std_mask(self.trg, pad)
                self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(trg, pad):
        "Create a mask to hide padding and future words."
        trg_mask = (trg != pad).unsqueeze(-2)
        trg_mask = trg_mask & sub_mask(trg.size(-1)).type_as(trg_mask.data)
        return trg_mask

    def collate_batch(
        self, batch, src_pipeline, trg_pipeline, src_vocab, trg_vocab, device, padding=128, pad_id=2
    ):
        bs_id = torch.tensor([0], device=device)  # <s> token ID
        eos_id = torch.tensor([1], device=device)  # </s> token ID
        src_list, trg_list = [], []

        for _src, _trg in batch:
            processed_src = torch.cat(
                [bs_id, torch.tensor(src_vocab(src_pipeline(_src)), dtype=torch.int64, device=device), eos_id],
                0,
            )
            processed_trg = torch.cat(
                [bs_id, torch.tensor(trg_vocab(trg_pipeline(_trg)), dtype=torch.int64, device=device), eos_id],
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
        self, device, vocab_src, vocab_trg, vocab2dec, vocab2enc, batch_sz=12000, padding=128, is_distributed=True
    ):
        # Tokenizer functions
        def tokenize_de(text):
            return Tokenizer.tokenize(text, vocab2dec)

        def tokenize_en(text):
            return Tokenizer.tokenize(text, vocab2enc)

        def collate_fn(batch):
            return self.collate_batch(
                batch, tokenize_de, tokenize_en, vocab_src, vocab_trg, device, padding=padding, pad_id=2
            )

        train_iter = datasets.Multi30k(split="train", language_pair=("de", "en"))
        valid_iter = datasets.Multi30k(split="valid", language_pair=("de", "en"))

        train_iter_map = to_map_style_dataset(train_iter)
        train_sampler = DistributedSampler(train_iter_map) if is_distributed else None
        valid_iter_map = to_map_style_dataset(valid_iter)
        valid_sampler = DistributedSampler(valid_iter_map) if is_distributed else None

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
https://github.com/pytorch/text/issues/2221
Might be issue with downloading test data.
"""

class Tokenizer:
    @staticmethod
    def get_tokens():
        try:
            vocab2dec = spacy.load("de_core_news_sm")
        except IOError:
            os.system("python -m spacy download de_core_news_sm")
            vocab2dec = spacy.load("de_core_news_sm")

        try:
            vocab2enc = spacy.load("en_core_web_sm")
        except IOError:
            os.system("python -m spacy download en_core_web_sm")
            vocab2enc = spacy.load("en_core_web_sm")

        return vocab2dec, vocab2enc

    @staticmethod
    def tokenize(text, tokenizer):
        return [tok.text for tok in tokenizer.tokenizer(text)]

    @staticmethod
    def yield_tokens(data_iter, tokenizer, index):
        for from_to_tuple in data_iter:
            yield tokenizer(from_to_tuple[index])

    @staticmethod
    def build_vocabulary(vocab2dec, vocab2enc):
        def tokenize_dec(text):
            return Tokenizer.tokenize(text, vocab2dec)

        def tokenize_enc(text):
            return Tokenizer.tokenize(text, vocab2enc)

        print("Building German Vocabulary ...")
        train, val, test = datasets.Multi30k(language_pair=("de", "en"))
        vocab_src = build_vocab_from_iterator(
            Tokenizer.yield_tokens(train + val + test, tokenize_dec, index=0),
            min_freq=2,
            specials=["<s>", "</s>", "<blank>", "<unk>"],
        )

        print("Building English Vocabulary ...")
        vocab_trg = build_vocab_from_iterator(
            Tokenizer.yield_tokens(train + val + test, tokenize_enc, index=1),
            min_freq=2,
            specials=["<s>", "</s>", "<blank>", "<unk>"],
        )
        vocab_src.set_default_index(vocab_src["<unk>"])
        vocab_trg.set_default_index(vocab_trg["<unk>"])

        return vocab_src, vocab_trg

    @staticmethod
    def get_vocab():
        if not exists("vocab.pt"):
            vocab2dec, vocab2enc = Tokenizer.get_tokens()
            vocab_src, vocab_trg = Tokenizer.build_vocabulary(vocab2dec, vocab2enc)
            torch.save((vocab_src, vocab_trg), "vocab.pt")
        else:
            vocab_src, vocab_trg = torch.load("vocab.pt")
        print("Finished.\nVocabulary sizes:")
        print(len(vocab_src))
        print(len(vocab_trg))
        return vocab_src, vocab_trg


# --- Loading Vocabulary --- #
# 1. Load tokenizers
# 2. Build vocabulary
# 3. To load tokens use get_vocab()

# --- Loading Dataset --- #
