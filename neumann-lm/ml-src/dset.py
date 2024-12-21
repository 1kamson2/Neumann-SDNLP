import numpy as np
import pandas as pd
import torch
import glob
import logging
from torch.utils.data import Dataset
from enum import Enum
from functools import lru_cache

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
    def __init__(self, data_flag, src_lang, trg_lang, src_emb, trg_emb):
        super().__init__()
        # Keep these for now, maybe for logging purpose and setup
        self.ed_data = {src_lang: src_emb, trg_lang: trg_emb}
        self.data_flag = data_flag
        self.data_path = ""
        self.bad_chars = []
        self.data_set = self.get_dataset()
        # --- DEBUGGING --- #

    @lru_cache(1)
    def get_dataset(self):
        """
        For now we can load the entire training data set. The biggest data
        set is only 1.3GB so it will fit into our memory, but maybe better
        approach would be partitioning our data set, even though it will surely
        fit
        """
        logging.basicConfig(
            filename="rsrc/nlp-dataset.log", encoding="utf-8", level=logging.DEBUG
        )
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

        logger.info("NLP DATASET IS BEING LOADED")
        dict_embed = []
        for translation in (
            row
            for row in pd.read_csv(
                self.data_path, lineterminator="\n", encoding="utf-8"
            )
            .head(32)
            .itertuples()
        ):

            idx, trg, src = translation
            if len(trg) <= 1 or len(src) <= 1:
                logger.info(
                    f"[WARNING] This input doesn't meet requirements, length is"
                    " less or equal than 1.\n"
                    f"Index: {idx} de: {trg} en: {src}\n"
                    f"Length: de: {len(trg)} en: {len(src)}\n"
                )
            else:
                logger.info(
                    f"Index: {idx}\n" f"Length: de: {len(trg)} en: {len(src)}\n"
                )
                src_proc = self.preprocess_text(src, self.ed_data.get(Language.SRC))
                trg_proc = self.preprocess_text(trg, self.ed_data.get(Language.TRG))
                add_pos_encoding(src_proc, self.ed_data.get(Language.SRC))
                add_pos_encoding(trg_proc, self.ed_data.get(Language.TRG))
                dict_embed.append(
                    dict(
                        source=torch.from_numpy(src_proc).unsqueeze(0),
                        target=torch.from_numpy(trg_proc).unsqueeze(0),
                    )
                )
        # -- Better way would be give a flag rather than store if doing this way
        # --
        if len(self.bad_chars) > 0:
            print(
                f"[WARNING] There have been bad characters in the dataset.\n"
                f"See logs."
            )
            logger.info(
                f"[WARNING] In your dataset have been characters "
                f"that didn't meet requirements:\n"
                f"[character] : [ascii]"
            )
            for el in self.bad_chars:
                logger.info(f"[{chr(el)}] : [{el}]")

        return np.array(dict_embed)

    def preprocess_text(self, text, embv):
        """
        Preprocess given text to an array of binaries. Embedded text array is
        not restricted to any hardcoded size. Length of word embedding depends
        on what we try to embed.
        """
        embedded_text = list()
        text = text.split(" ")
        # subs_char = {chr(8217): "'", chr(8222): '"', chr(8220): '"'}
        for word in text:
            for c in word:
                ec = [0] * embv
                i = 0
                if ord(c) > 128 and ord(c) not in self.bad_chars:
                    self.bad_chars.append(ord(c))

                for b in bin(ord(c))[2::]:
                    try:
                        if b == "1":
                            ec[i] = 1
                        i += 1
                        embedded_text.append(ec)
                    except IndexError as e:
                        logger.error(
                            f"Error occurred: {e}\nThis error occurred for:",
                            f"[word:] [{word}] [character:] [{c}] [bin:]",
                            f"[{bin(ord(c))}] [ascii:] [{ord(c)}]",
                        )
                        exit(-1)
        padding = 4096 - len(embedded_text)
        if padding < 0:
            logger.error("Padding is less than 0")
            exit(-1)
        else:
            to_pad = [[0] * embv] * padding
            embedded_text.extend(to_pad)
        return np.array(embedded_text, dtype=np.float32)

    def __getitem__(self, index):
        return self.data_set[index]

    def __len__(self):
        return len(self.data_set)
