
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, num_inputs, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
