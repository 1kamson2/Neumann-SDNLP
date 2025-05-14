from typing import List, Tuple
import spacy
from spacy.vocab import Language
from datasets import load_dataset
def get_vocabs() -> Tuple[Language, Language]:
  """
    Load English Vocabulary and German Vocabulary. The English vocabulary will
    be encoded and the German vocabulary will be decoded. 

    Returns:
      Vocabularies in the following order:
        German, English.
  """
  de_vocab: Language = spacy.load("de_core_news_sm")
  en_vocab: Language = spacy.load("en_core_web_sm")
  return de_vocab, en_vocab

def tokenize(content: str, vocab: Language) -> List:
  """
    Tokenize the text using the given vocabulary.

    Arguments:
      content: The text that will be tokenized.
      vocab: The vocabulary that will be used to tokenize the text.

    Returns:
      List of tokens.
  """
  return [tok.text for tok in vocab.tokenizer(content)]

def yield_tokens(all_toks, vocab, index):
  for from_to_tuple in all_toks:
    yield vocab(from_to_tuple[index])


def build_final_vocabs(vocab_decoder: Language, vocab_encoder: Language):
  """
    Build the source and target vocabularies that will be used in training the
    model.

    Arguments:
      vocab_decoder: The vocabulary that will be used for decoding.
      vocab_encoder: The vocabulary that will be used for encoding.

    Returns:
      ???
  """
  # TODO: REMOVE
  # def tokenize_dec(text):
  #   return Tokenizer.tokenize(text, vocab2dec)
  #
  # def tokenize_enc(text):
  #   return Tokenizer.tokenize(text, vocab2enc)

  # No split yields to DatasetDict
  data_dict = load_dataset("bentrevett/multi30k") 

  # TODO: Find out what does the adding of all sets do.
  vocab_src = build_vocab_from_iterator(
    yield_tokens(train, vocab_decoder, index=0),
    min_freq=2,
    specials=["<s>", "</s>", "<blank>", "<unk>"],
  )
  logger.info("German vocabulary built.")
  vocab_tgt = build_vocab_from_iterator(
    yield_tokens(train + validation + test, tokenize_enc, index=1),
    min_freq=2,
    specials=["<s>", "</s>", "<blank>", "<unk>"],
  )
  logger.info("English vocabulary built.")
  vocab_src.set_default_index(vocab_src["<unk>"])
  vocab_tgt.set_default_index(vocab_tgt["<unk>"])

  return vocab_src, vocab_tgt

def get_vocab():
  VOCABULARY_PATH = FileManager().python_data_catalog
  if not exists(VOCABULARY_PATH):
    logger.info(f"{VOCABULARY_PATH} not detected. Downloading...")
    vocab2dec, vocab2enc = Tokenizer.get_tokens()
    vocab_src, vocab_trg = Tokenizer.build_vocabulary(vocab2dec, vocab2enc)
    torch.save((vocab_src, vocab_trg), FileManager().python_data)
    logger.info(f"{VOCABULARY_PATH} saved.")
  else:
    vocab_src, vocab_trg = torch.load(VOCABULARY_PATH)
    logger.info(f"{VOCABULARY_PATH} loaded.")

  logger.info("Vocabulary has been created:")
  logger.info(f"-- Source vocabulary: {len(vocab_src)}")
  logger.info(f"-- Target vocabulary: {len(vocab_trg)}")
  return vocab_src, vocab_trg

