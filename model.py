import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class MouseCommsNet(nn.Module):
    def __init__(self):
        super(MouseCommsNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(80, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layer1(x)
        return x
