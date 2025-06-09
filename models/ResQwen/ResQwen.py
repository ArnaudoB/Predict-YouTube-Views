from models.ResQwen.ResNetEncoder import ResNetEncoder
from models.ResQwen.FusionMLP import FusionMLP
from models.ResQwen.DateMLP import DateMLP
from models.ResQwen.ChannelMLP import ChannelMLP
from models.ResQwen.QwenEncoder import QwenEncoder
from models.ResQwen.constants import n_fusion, nb_temp_features, nb_channels
import torch.nn as nn
import torch

class ResQwen(nn.Module):
    def __init__(self, dropout=0.3, semifreeze_qwen=False, semifreeze_resnet=False, freeze_qwen=True, freeze_resnet=True):
        super(ResQwen, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.resnet = ResNetEncoder(output_dim=128, dropout=dropout, frozen=freeze_resnet, semifrozen=semifreeze_resnet).to(device)
        self.fusion_mlp = FusionMLP(input_dim=n_fusion, output_dim=1, dropout=dropout).to(device)
        self.date_mlp = DateMLP(input_dim=nb_temp_features, output_dim=32, dropout=dropout).to(device)
        self.channel_mlp = ChannelMLP(input_dim=nb_channels, output_dim=64, dropout=dropout).to(device)
        self.qwen = QwenEncoder(output_dim=128, dropout=dropout, frozen=freeze_qwen, semifrozen=semifreeze_qwen)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        # x is a dictionary with keys 'img', 'date', 'channel', 'title'

        img = self.resnet(x['image'])
        date = self.date_mlp(x['date'])
        channel = self.channel_mlp(x['channel'])
        title = self.qwen(x['title'])

        # Move all tensors to the same device
        img = img.to(date.device)

        # Concatenate the features
        fusion = torch.cat((img, title, channel, date), dim=1)
        fusion = self.dropout(fusion)
        fusion = self.fusion_mlp(fusion)

        return fusion