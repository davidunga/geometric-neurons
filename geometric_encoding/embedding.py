import torch
from abc import ABC, abstractmethod


class Embedder(ABC, torch.nn.Module):

    def __init__(self, input_size: int, output_size: int, *args, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    @abstractmethod
    def forward(self, x):
        pass


class LinearEmbedder(Embedder):

    def __init__(self, input_size: int, output_size: int, dropout: float):
        super().__init__(input_size, output_size)
        torch.manual_seed(0)
        self.embedder = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(self.input_size, self.output_size))

    def forward(self, x):
        return self.embedder(x)
