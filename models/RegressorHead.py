import torch.nn as nn
from constants import n_features
# n_features = 1873

class RegressorHead(nn.Module):
    def __init__(self, input_dim=n_features, output_dim=1, dropout=0.1):
        super(RegressorHead, self).__init__()
        self.regression_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), # 1873 -> 936
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(input_dim // 2, input_dim // 4), # 936 -> 468
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(input_dim // 4, input_dim // 8), # 468 -> 234
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(input_dim // 8, input_dim//16), # 234 -> 117
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(input_dim // 16, output_dim) # 117 -> 1
        )

    def forward(self, x):
        return self.regression_head(x)