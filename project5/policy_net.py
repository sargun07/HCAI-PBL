import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # up/down/left/right
            nn.Softmax(dim=-1)
        )

    def forward(self, x):  # x shape: [batch_size, 6, 5, 5]
        return self.model(x)
