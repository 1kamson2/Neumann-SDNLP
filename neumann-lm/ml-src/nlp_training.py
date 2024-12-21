from nns import NLPEncoder, Transformer
from dset import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim.sgd import SGD


class NLP:
    def __init__(self, lr=0.001, momentum=0.99):
        self.sth = "Something should be here"
        self.lr = lr
        self.momentum = momentum
        self.model = Transformer(16, 16)
        self.optimizer = SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum
        )
        self.dataset = NLPDataset(
            DataType.TRAINING,
            Language.SRC,
            Language.TRG,
            EN_WORD_EMBEDDING_LENGTH,
            DE_WORD_EMBEDDING_LENGTH,
        )
        # [FIX] Shit doesnt work if batch size is bigger than 1
        # self.batch_sz = 32
        # The problem with ex which after encoder is of size [248, 1, ...]
        self.batch_sz = 1

    def training(self, nepoch):
        dl = DataLoader(dataset=self.dataset, batch_size=self.batch_sz, shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        # nbatch = 0
        for epoch in range(nepoch):
            for data in dl:
                print(data["target"].shape, data["source"].shape)
                pred = self.model(data["target"], data["source"])
                print(pred.shape, pred)
                print(
                    f"Predicting: {data["target"]}\nTarget: {data["source"]}\n"
                    f"Predicted: {pred.shape} {pred}"
                )
                print(data["target"].shape)
                loss = loss_fn(pred, data["target"])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
