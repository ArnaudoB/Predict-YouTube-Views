from models.ResBert.ResNetEncoder import ResNetEncoder
from models.ResBert.FusionMLP import FusionMLP
from models.ResBert.DateMLP import DateMLP
from models.ResBert.ChannelMLP import ChannelMLP
from models.ResBert.BertEncoder import BertEncoder
from models.ResBert.constants import n_fusion, nb_temp_features, nb_channels
import torch.nn as nn
import torch

class ResBert(nn.Module):
    def __init__(self, dropout=0.3, semifreeze_bert=False, semifreeze_resnet=False, freeze_bert=True, freeze_resnet=True):
        super(ResBert, self).__init__()
        self.resnet = ResNetEncoder(output_dim=256, dropout=dropout, frozen=freeze_resnet, semifrozen=semifreeze_resnet)
        self.fusion_mlp = FusionMLP(input_dim=n_fusion, output_dim=1, dropout=dropout)
        self.date_mlp = DateMLP(input_dim=nb_temp_features, output_dim=256, dropout=dropout)
        self.channel_mlp = ChannelMLP(input_dim=nb_channels, output_dim=256, dropout=dropout)
        self.bert = BertEncoder(output_dim=256, dropout=dropout, frozen=freeze_bert, semifrozen=semifreeze_bert)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is a dictionary with keys 'img', 'date', 'channel', 'title'
        img = self.resnet(x['image'])
        date = self.date_mlp(x['date'])
        channel = self.channel_mlp(x['channel'])
        title = self.bert(x['title'])

        # Concatenate the features
        fusion = torch.cat((img, title, channel, date), dim=1)
        fusion = self.dropout(fusion)
        fusion = self.fusion_mlp(fusion)

        return fusion