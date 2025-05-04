import torch.nn as nn

class RegressorHead(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(RegressorHead, self).__init__()
        self.regression_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, output_dim)
        )

    def forward(self, x):
        return self.regression_head(x)