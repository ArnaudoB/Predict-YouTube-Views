import torch.nn as nn
from models.ClipBertViewPredictor.constants import n_features

class RegressorHead(nn.Module):
    def __init__(self, input_dim=n_features, output_dim=1, dropout=0.3):
        super(RegressorHead, self).__init__()
        self.regression_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), # 542 -> 271
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(input_dim // 2, input_dim // 4), # 271 -> 135
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(input_dim // 4, input_dim // 8), # 135 -> 67
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(input_dim // 8, output_dim) # 67 -> 1
            
        )

    def forward(self, x):
        return self.regression_head(x)