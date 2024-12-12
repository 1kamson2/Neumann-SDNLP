import numpy as np
import pandas as pd
import torch
import glob
from torch.utils.data import Dataset
from enum import Enum
from functools import lru_cache

NLP_TRAINING_DATA_PATH = "rsrc/WMT14_train.csv"
NLP_VALIDATION_DATA_PATH = "rsrc/WMT14_validation.csv"
NLP_TEST_DATA_PATH = "rsrc/WMT14_test.csv"
EN_WORD_EMBEDDING_LENGTH = 7
DE_WORD_EMBEDDING_LENGTH = 8


class DataType(Enum):
    TRAINING = 0
    VALIDATION = 1
    TEST = 2


class Language(Enum):
    EN = "en"
    DE = "de"


def add_pos_encoding(embedded_text, embed) -> None:
    d_model = embed
    is_odd = d_model % 2 != 0
    n = 10000
    # How much embedding, length of each embedding
    cnt_emb, len_emb = embedded_text.shape
    for pos in range(cnt_emb):
        pos_enc = [0] * len_emb
        for i in range(d_model // 2):
            pos_enc[2 * i] = np.sin(pos / n ** (2 * i / d_model))
            pos_enc[2 * i + 1] = np.cos(pos / n ** (2 * i / d_model))
        if is_odd:
            pos_enc[d_model - 1] = np.sin(pos / n ** (2 * (d_model - 1) / d_model))
        pos_enc = np.array(pos_enc, dtype=np.float32)
        embedded_text[pos] += pos_enc


class NLPDataset(Dataset):
    def __init__(self, data_flag):
        super().__init__()
        # Keep these for now, maybe for logging purpose and setup
        self.data_flag = data_flag
        self.data_path = ""
        self.data_set = self.get_dataset()

    @lru_cache(1)
    def get_dataset(self):
        """
        For now we can load the entire training data set. The biggest data
        set is only 1.3GB so it will fit into our memory, but maybe better
        approach would be partitioning our data set, even though it will surely
        fit
        """
        assert len(glob.glob("rsrc/*.csv")) == 3, (
            f"You are probably missing one or more "
            f"of the files:"
            f"\n{NLP_TRAINING_DATA_PATH}\n{NLP_TEST_DATA_PATH}\n"
            f"{NLP_VALIDATION_DATA_PATH}"
        )
        match self.data_flag:
            case DataType.TRAINING:
                self.data_path = NLP_TRAINING_DATA_PATH
            case DataType.VALIDATION:
                self.data_path = NLP_VALIDATION_DATA_PATH
            case DataType.TEST:
                self.data_path = NLP_TEST_DATA_PATH
        assert (
            len(self.data_path) != 0 and "Not implemented" not in self.data_path
        ), "Assertion failed: Neumann couldn't load your data set."

        print("########## LOADING DATA ##########")
        iembedding = []
        for translation in (
            row for row in pd.read_csv(self.data_path, lineterminator="\n").itertuples()
        ):

            idx, de, en = translation
            if len(de) <= 1 or len(en) <= 1:
                print(
                    f"[WARNING] This input doesn't meet requirements, length is"
                    " less or equal than 1.\n"
                    f"Index: {idx} de: {de} en: {en}\n"
                    f"Length: de: {len(de)} en: {len(en)}\n"
                )
            else:
                print(f"Index: {idx}\n" f"Length: de: {len(de)} en: {len(en)}\n")
                en_proc = self.preprocess_text(en, EN_WORD_EMBEDDING_LENGTH)
                de_proc = self.preprocess_text(en, DE_WORD_EMBEDDING_LENGTH)
                add_pos_encoding(en_proc, EN_WORD_EMBEDDING_LENGTH)
                add_pos_encoding(de_proc, DE_WORD_EMBEDDING_LENGTH)
                iembedding.append(
                    dict(en=torch.from_numpy(en_proc), de=torch.from_numpy(de_proc))
                )

        return np.array(iembedding)

    def preprocess_text(self, text, emb_type):
        """
        Preprocess given text to an array of binaries. Embedded text array is
        not restricted to any hardcoded size. Length of word embedding depends
        on what we try to embed.
        """
        # todo: weird out of range index error.
        embedded_text = list()
        text = text.split(" ")
        for word in text:
            try:
                for c in word:
                    ec = [0] * emb_type
                    i = 0
                    for b in bin(ord(c))[2::]:
                        if b == "1":
                            ec[i] = 1
                        i += 1
                    embedded_text.append(ec)
            except IndexError as e:
                print(e)
        return np.array(embedded_text, dtype=np.float32)

    def __getitem__(self, index):
        return self.data_set[index]

    def __len__(self):
        return len(self.data_set)
