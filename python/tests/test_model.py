import random
from models.nlp.model import Embeddings, deep_copy_module

def test_cloning() -> None:
  """
    Test if copying modules work correctly.
  """
  rand_d_model: int = random.randint(1, 100)
  rand_d_vocab: int = random.randint(1, 100)
  n: int = random.randint(1, 50)
  modules_cloned = deep_copy_module(Embeddings, n, rand_d_model, rand_d_vocab)
  assert len(modules_cloned) == n

