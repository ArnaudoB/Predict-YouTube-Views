import torch.nn as nn
from models.ResQwen.constants import n_fusion
from torch.nn import init

class FusionMLP(nn.Module):
    def __init__(self, input_dim=n_fusion, output_dim=1, dropout=0.2):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(128),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(64),
            nn.Dropout(dropout),

            nn.Linear(64, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, a=0.01, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        return self.mlp(x)
    