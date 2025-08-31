import torch, torch.nn as nn

class RewardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(16*5*5, 64), nn.ReLU(),
            nn.Linear(64, 1)  # per-state reward scalar
        )
    def forward(self, s):  # s: [B,6,5,5]
        z = self.conv(s)
        r = self.head(z)
        return r.squeeze(-1)  # [B]
