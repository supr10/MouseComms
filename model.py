import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class MouseCommsNet(nn.Module):
    def __init__(self):
        super(MouseCommsNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(40, 64),
            nn.Softplus(),
            nn.Linear(64, 128),
            nn.Softplus(),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 32),
            nn.Softplus(),
            nn.Linear(32, 5),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layer1(x)
        return x
