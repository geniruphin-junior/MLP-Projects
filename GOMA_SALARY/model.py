import torch
import torch.nn as nn

# creation du model


class NetworkNeural(nn.Module):
    def __init__(self, n_feactures: torch.Tensor):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(n_feactures, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor):
        x = self.sequence(x)
        return x
