from RegressorHead import RegressorHead
from ClipBertEncoder import ClipBertEncoder
import torch
import torch.nn as nn
from constants import n_features

class ClipBertViewPredictor(nn.Module):
    def __init__(self, frozen=False):
        super().__init__()
        self.clipbert_encoder = ClipBertEncoder(frozen=frozen)
        self.regression_head = RegressorHead(input_dim=n_features, output_dim=1, dropout=0.1)

    def forward(self, x):
        # Get the features from the encoder
        features = self.clipbert_encoder(x)
        # Concatenate the features from image, title, and description
        midput = torch.cat([features, x["tabular"]], dim=1)  # Concatenate along the feature dimension
        # Pass the features through the regression head
        output = self.regression_head(midput)
        
        return output
    
# if __name__ == '__main__':
#     model = MultimodalCLIPRegressor(frozen=True)
#     output = model({
#         "image": processed_image_tensor,
#         "title": ["Video title here"],
#         "description": ["Longer description of the video content..."],
#         "tabular": tabular_data_tensor
#     })