import torch.nn as nn
from torchvision import models

class ResNetEncoder(nn.Module):
    def __init__(self, output_dim=256, dropout=0.2, frozen=True, semifrozen=False):
        super(ResNetEncoder, self).__init__()

        self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        
        # Remove the final fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # Add a global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.img_dim = 2048  # ResNet50 output dimension after global average pooling

        if frozen:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Unfreeze the last layer of ResNet if semifrozen is True
        elif semifrozen:
            for name, param in self.resnet.named_parameters():
                if name.startswith('7.'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        # Projection layer to reduce dimensions
        self.projection_model = nn.Sequential(nn.Linear(self.img_dim, output_dim),
                                                  nn.LeakyReLU(0.1),
                                                  nn.LayerNorm(output_dim),
                                                  nn.Dropout(dropout))
        
        self._init_weights()

    def _init_weights(self):

        for layer in self.projection_model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, a=0.1, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        
    def forward(self, x):

        # Pass the input through the ResNet model
        x = self.resnet(x)
        
        # Apply global average pooling
        x = self.global_avg_pool(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)

        # Pass through the projection model
        x = self.projection_model(x)
        
        return x