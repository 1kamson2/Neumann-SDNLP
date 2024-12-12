from nns import NLPEncoder
from dset import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


class NLP:
    def __init__(self, lr=0.001, momentum=0.99):
        self.sth = "Something should be here"
        self.lr = lr
        self.momentum = momentum
        self.model = NLPEncoder()
        self.dataset = NLPDataset(DataType.TRAINING)
        self.batch_sz = 32

    def training(self, nepoch):
        dl = DataLoader(dataset=self.dataset, batch_size=self.batch_sz, shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        # nbatch = 0
        for epoch in range(nepoch):
            for data in dl:
                pred = self.model(data["en"])
                loss = loss_fn(pred, data["de"])
