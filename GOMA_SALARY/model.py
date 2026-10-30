import torch
import torch.nn as nn

# creation du model


class NetworkNeural(nn.Module):
    def __init__(self, n_feactures: torch.Tensor, n_classes, dropout=0.1):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(n_feactures, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.sequence(x)
        return x
