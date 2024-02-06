import torch
from abc import ABC, abstractmethod


class Embedder(ABC, torch.nn.Module):

    def __init__(self, input_size: int, dim: int, *args, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.dim = dim

    @abstractmethod
    def forward(self, x):
        pass


class LinearEmbedder(Embedder):

    def __init__(self, input_size: int, dim: int, dropout: float, init_seed: int = 0):
        super().__init__(input_size, dim)
        torch.manual_seed(init_seed)
        self.embedder = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(self.input_size, self.dim))

    def forward(self, x):
        return self.embedder(x)


