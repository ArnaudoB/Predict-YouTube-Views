import torch.nn as nn
from models.ResBert.constants import nb_channels

class ChannelMLP(nn.Module):
    def __init__(self, input_dim=nb_channels, output_dim=256, dropout=0.3):
        super(ChannelMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.mlp(x)