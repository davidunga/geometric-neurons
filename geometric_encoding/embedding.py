import torch

torch.manual_seed(0)


def embdded_dist_fnc(x1, x2):
    return torch.sum((x1 - x2) ** 2, dim=1)


class LinearEmbedder(torch.nn.Module):

    def __init__(self, input_size, descriptor_size, dropout):
        super(LinearEmbedder, self).__init__()
        self.input_size = input_size
        self.decision_thresh = None
        self.embedder = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(self.input_size, descriptor_size))

    def forward(self, x):
        return self.embedder(x)
