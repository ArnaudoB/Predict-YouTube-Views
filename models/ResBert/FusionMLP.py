import torch.nn as nn
from models.ResBert.constants import n_fusion

class FusionMLP(nn.Module):
    def __init__(self, input_dim=n_fusion, output_dim=1, dropout=0.3):
        super(FusionMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim//2), # n_fusion = 256*4 -> 256*2
            nn.ReLU(),
            nn.LayerNorm(input_dim//2),
            nn.Dropout(dropout),


            nn.Linear(256*2, 128), # 256*2 -> 128
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),

            nn.Linear(128, 64), # 128 -> 64
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),

            nn.Linear(64, 1), # 64 -> 1
            nn.ReLU()
            )
    

        
    def forward(self, x):
        return self.mlp(x)
    