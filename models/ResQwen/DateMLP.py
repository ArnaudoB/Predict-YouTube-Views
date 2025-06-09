import torch.nn as nn
from models.ResQwen.constants import nb_temp_features

class DateMLP(nn.Module):
    def __init__(self, input_dim=nb_temp_features, output_dim=128, dropout=0.2):
        super(DateMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout))
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, a=0.1, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        return self.mlp(x)