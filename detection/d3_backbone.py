import torch
from torch import nn
from torch.nn.functional import layer_norm, relu
from einops import rearrange

from core.settings import model_config

class Detect3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = model_config.input_dim
        self.dim = model_config.dim
        self.bev_num = model_config.bev_num
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(self.input_dim, 32, kernel_size=3, padding='same'),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm3d(32)
        )
        self.conv3d_2 =nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm3d(64),
        )
        self.conv3d_3 =nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=5, padding='same'),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=5, padding='same'),
            nn.BatchNorm3d(64),
        )
        self.conv1d_1 = nn.Conv3d(self.input_dim, 32, kernel_size=1, padding='same')
        self.conv1d_2 = nn.Conv3d(32, 64, kernel_size=1, padding='same')
        self.conv1d_3 = nn.Conv3d(64, 64, kernel_size=1, padding='same')
        self.maxpool_1 = nn.MaxPool3d(2)
        self.maxpool_2 = nn.MaxPool3d(2)
        self.maxpool_3 = nn.MaxPool3d(2)


    def forward(self, x):

        x = self.resnet(x)
        x = layer_norm(x, [x.size()[-4], x.size()[-3], x.size()[-2], x.size()[-1]])
        out = self.bev_map(x)

        return out
    
    def bev_map(self, x):
        out = rearrange(x, 'b c dy dx dz -> b (dy c) dx dz')
        
        return out
    
    def resnet(self, x):
        x_conv3d_1 = self.conv3d_1(x)
        x_maxpool_1 = self.maxpool_1(relu(x_conv3d_1 + self.conv1d_1(x)))

        x_conv3d_2 = self.conv3d_2(x_maxpool_1)
        x_maxpool_2 = self.maxpool_2(relu(x_conv3d_2 + self.conv1d_2(x_maxpool_1)))

        x_conv3d_3 = self.conv3d_3(x_maxpool_2)
        x_maxpool_3 = self.maxpool_3(relu(x_conv3d_3 + self.conv1d_3(x_maxpool_2)))

        return x_maxpool_3
    