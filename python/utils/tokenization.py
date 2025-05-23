from typing import Iterator, List, Tuple
import spacy
from spacy.language import Language
from datasets import Dataset, DatasetDict, load_dataset
from torchtext.vocab import Vocab, build_vocab_from_iterator 

SPECIAL_TOKS: List = ["<s>", "</s>", "<blank>", "<unk>"]
def get_tokenizers() -> Tuple[Language, Language]:
  """
    Load English and German Tokenizer. 
    Returns:
      German Tokenizer, English Tokenizer.

  """
  nlp_de: Language = spacy.load("de_core_news_sm")
  nlp_en: Language = spacy.load("en_core_web_sm")
  return nlp_de, nlp_en

def tokenize(content: str, nlp: Language) -> List:
  """
    Tokenize the text using the given vocabulary.

    Arguments:
      content: The text that will be tokenized.
      nlp: The Tokenizer that will be used to tokenize the text.

    Returns:
      List of tokens.
  """
  return [tok.text for tok in nlp.tokenizer(content)]

def yield_tokens(toks: List, nlp: Language) -> Iterator:
  """
    Build the generator from tokens.
    
    Arguments:
      toks: List of tokens.
      nlp: Tokenizer to tokenize.

    Returns:
      Generator of tokens.
  """
  for tok_str in toks:
    yield nlp(tok_str)


def build_final_vocabs(
  nlp_de: Language, 
  nlp_en: Language, 
  should_serialize: bool = True
) -> Tuple[Vocab, Vocab]:
  """
    Build the source and target vocabularies that will be used in training the
    model.

    Arguments:
      nlp_de: The vocabulary that will be used for decoding.
      nlp_en: The vocabulary that will be used for encoding.

    Returns:
      English Vocabulary and German Vocabulary.
  """

  # No split yields to DatasetDict
  dataset_pair = load_dataset("bentrevett/multi30k") 

  assert isinstance(dataset_pair, DatasetDict)
  train: Dataset = dataset_pair["train"] 
  # TODO: Find a way to add those datasets.
  validation: Dataset = dataset_pair["validation"] 
  test: Dataset = dataset_pair["test"] 

  # TODO: Find out what does the adding of all sets do.
  vocab_src = build_vocab_from_iterator(
    yield_tokens(train["en"], nlp_de),
    min_freq=2,
    specials=SPECIAL_TOKS,
  )

  vocab_tgt = build_vocab_from_iterator(
    yield_tokens(train["de"], nlp_en),
    min_freq=2,
    specials=SPECIAL_TOKS,
  )

  vocab_src.set_default_index(vocab_src["<unk>"])
  vocab_tgt.set_default_index(vocab_tgt["<unk>"])

  if should_serialize:
    serialize_vocabulary(vocab_src)
    serialize_vocabulary(vocab_tgt)
  return vocab_src, vocab_tgt


def serialize_vocabulary(vocabulary: Vocab) -> None:
  """
    Serialize the vocabulary. 

    Arguments:
      vocabulary: The vocabulary to be saved.
  """
  

def load_local_vocabulary() -> Tuple[Vocab, Vocab]:
  """
    Load vocabularies from the catalog.
  """
  assert 1 == 2, "NOT IMPLEMENTED"
  VOCABULARY_PATH = FileManager().python_data_catalog
  if not exists(VOCABULARY_PATH):
    logger.info(f"{VOCABULARY_PATH} not detected. Downloading...")
    vocab2dec, vocab2enc = get_tokenizers()
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

