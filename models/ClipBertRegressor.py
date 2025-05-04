from models.RegressorHead import RegressorHead
from models.ClipBertEncoder import ClipBertEncoder
import torch.nn as nn

class ClipBertRegressor(nn.Module):
    def __init__(self, frozen=False):
        super().__init__()
        self.clipbert_encoder = ClipBertEncoder(frozen=frozen)
        self.regression_head = RegressorHead(self.clipbert_encoder.img_dim * 3)

    def forward(self, x):
        # Get the features from the encoder
        features = self.clipbert_encoder(x)
        
        # Pass the features through the regression head
        output = self.regression_head(features)
        
        return output
    
# if __name__ == '__main__':
#     model = MultimodalCLIPRegressor(frozen=True)
#     output = model({
#         "image": processed_image_tensor,
#         "title": ["Video title here"],
#         "description": ["Longer description of the video content..."]
#     })