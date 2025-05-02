from nlp_model.nlp import *
from dset import FileManager, Tokenizer
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import time
import logging

logger = logging.getLogger(__name__)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NLP:
  def __init__(self, **config):
    """
    There should be easier and faster way to copy layers / networks in
    torch, because deep copy is really expensive operation and very slow.
    """
    # --- Initialize parameters for the batch training --- #
    # 1. Get vocabs.
    vocab_src, vocab_trg = Tokenizer.get_vocab()
    # 2. Get lens.
    voclen_src, voclen_trg = len(vocab_src), len(vocab_trg)
    # 3. Get tokens.
    vocab2dec, vocab2enc = Tokenizer.get_tokens()  # (trg, src)

    # --- Initialize Layers for Transformer --- #
    print(
      "Transfomer started initialization. For more info about model"
      "parameters check help."
    )
    logger.info(
      "Transfomer started initialization. For more info about model"
      "parameters check help."
    )
    c = copy.deepcopy
    self.attn = MultiHeadAttention(config["h"], config["dmodel"]).to(_device)
    self.ffn = FeedForwardNetwork(
      config["dmodel"], config["dffn"], config["dropout"]
    ).to(_device)
    self.pos = PositionalEncoding(config["dmodel"], config["dropout"]).to(
      _device
    )
    self.generator = Generator(config["dmodel"], voclen_trg)
    self.padding_idx = vocab_trg["<blank>"]
    self.model = Transformer(
      NLPEncoder(
        EncoderLayer(
          config["dmodel"], c(self.attn), c(self.ffn), config["dropout"]
        ),
        config["N"],
      ),
      NLPDecoder(
        DecoderLayer(
          config["dmodel"],
          c(self.attn),
          c(self.attn),
          c(self.ffn),
          config["dropout"],
        ),
        config["N"],
      ),
      nn.Sequential(Embeddings(config["dmodel"], voclen_src), c(self.pos)),
      nn.Sequential(Embeddings(config["dmodel"], voclen_trg), c(self.pos)),
      self.generator,
    ).to(_device)
    # --- Initialize variables for epoch tracking --- #
    self.nepoch = 32
    self.warmup = 400
    self.factor = 1.0
    self.step = 0
    self.accum_step = 0
    self.samples = 0
    self.tokens = 0
    self.factor = 0
    self.n_accum = 0

    # --- Do src=None, if you don't want to generate the random batch --- #
    self.batch_sz = 12000
    self.batch = Batch(None, None, finit=True)
    self.optimizer = Adam(
      self.model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
    )
    self.scheduler = LambdaLR(
      optimizer=self.optimizer, lr_lambda=lambda _: self.rate()
    )
    self.criterion = LabelSmoothing(voclen_src, self.padding_idx, -1.1).to(
      _device
    )
    self.loss_fn = Loss(self.generator, self.criterion)
    self.t_dl, self.v_dl = self.batch.get_dataloader(
      device=_device,
      vocab_src=vocab_src,
      vocab_trg=vocab_trg,
      vocab2dec=vocab2dec,
      vocab2enc=vocab2enc,
      batch_sz=1024,
      padding=128,
      is_distributed=False,
    )
    self.transformer_weights = FileManager().transformer_weights_path
    print("Transformer initialization done.")
    logger.info("Transformer initialization done.")

  def rate(self):
    if self.step <= 0:
      self.step = 1
    return self.factor * (
      512 ** (-0.5)
      * min(self.step ** (-0.5), self.step * self.warmup ** (-1.5))
    )

  def run_training(self, mode="train", accum_iter=1):
    print("Transformer currently running training.")
    logger.info("Transformer currently running training.")
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(self.t_dl):
      _batch = Batch(batch[0], batch[1])
      out = self.model.forward(
        _batch.src, _batch.trg, _batch.src_mask, _batch.trg_mask
      )
      loss, loss_node = self.loss_fn(out, _batch.trg_y, _batch.ntokens)
      # loss_node = loss_node / accum_iter
      if mode == "train" or mode == "train+log":
        loss_node.backward()
        self.step += 1
        self.samples += _batch.src.shape[0]
        self.tokens += _batch.ntokens
        if i % accum_iter == 0:
          self.optimizer.step()
          self.optimizer.zero_grad(set_to_none=True)
          self.n_accum += 1
          self.accum_step += 1
        self.scheduler.step()

      total_loss += loss
      total_tokens += _batch.ntokens
      tokens += _batch.ntokens
      if i % 40 == 1 and (mode == "train" or mode == "train+log"):
        self.scheduler = self.optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start
        try:
          msg_info = (
            f"Epoch Step: {i} | Accumulation Step: {n_accum} | "
            + f"Loss: {loss / _batch.ntokens: 6.2f} | "
            + f"Tokens/s: {tokens / elapsed: 7.1f} | LR: "
            + f"{self.scheduler: 6.1f}"
          )
          print(msg_info)
          logger.info(msg_info)
        except IOError:
          print("Problem occurred while trying to log info about" + " epoch")
          logger.info(
            "Problem occurred while trying to log info" + " about epoch"
          )
          continue

        start = time.time()
        tokens = 0
    return total_loss / total_tokens
