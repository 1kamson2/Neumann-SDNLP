import torch
import torch.nn as nn
from enum import Enum


class NLPModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.init_model()

    # Implementing everything from scratch.
