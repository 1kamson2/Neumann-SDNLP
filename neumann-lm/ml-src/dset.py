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

        # We are looking for output of tuples: iembedding = [(..., ...), ...]
        iembedding = []
        for translation in (
            row
            for row in pd.read_csv(self.data_path, lineterminator="\n")
            .head(10)
            .itertuples()
        ):

            idx, de, en = translation
            if len(de) <= 1 or len(en) <= 1:
                print(
                    f"[WARNING] This input doesn't meet requirements, length is"
                    " less or equal than 1.\n"
                    f"Index:  {idx}\nde: {de}\nen: {en}\n"
                    f"Length: \nde: {len(de)}\nen: {len(en)}\n"
                )
            else:
                print(
                    f"Index: {idx}\nde: {de}\nen: {en}\n"
                    f"Length: \nde: {len(de)}\nen: {len(en)}\n"
                )
                en_proc = self.preprocess_text(en, EN_WORD_EMBEDDING_LENGTH)
                de_proc = self.preprocess_text(en, DE_WORD_EMBEDDING_LENGTH)
                iembedding.append((en_proc, de_proc))
        # Return doesn't work for now.
        return np.array(iembedding)

    def preprocess_text(self, text, emb_type):
        """
        Preprocess given text to an array of binaries. Embedded text array is
        not restricted to any hardcoded size. Length of word embedding depends
        on what we try to embed.
        """
        embedded_text = list()
        text = text.split(" ")
        for word in text:
            for c in word:
                ec = [0] * emb_type
                i = 0
                for b in bin(ord(c))[2::]:
                    if b == "1":
                        ec[i] = 1
                    i += 1
                embedded_text.append(ec)
        return np.array(embedded_text, dtype=np.int32)

    def __getitem__(self, index):
        return self.data_set[index]

    def __len__(self):
        return len(self.data_set)
