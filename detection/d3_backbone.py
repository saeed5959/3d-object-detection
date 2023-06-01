import torch
from torch import nn
from torch.nn.functional import layer_norm, relu
from einops import rearrange

from core.settings import model_config, train_config

device = train_config.device

class Detect3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = model_config.input_dim
        self.dim = model_config.dim
        self.bev_num = model_config.bev_num
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(self.input_dim, 16, kernel_size=3, padding='same'),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding='same'),
            nn.BatchNorm3d(32)
        )
        self.conv3d_2 =nn.Sequential(
            nn.Conv3d(32, 48, kernel_size=3, padding='same'),
            nn.BatchNorm3d(48),
            nn.ReLU(),
            nn.Conv3d(48, 64, kernel_size=3, padding='same'),
            nn.BatchNorm3d(64),
        )
        self.conv1d_1 = nn.Conv3d(self.input_dim, 32, kernel_size=1, padding='same')
        self.conv1d_2 = nn.Conv3d(32, 64, kernel_size=1, padding='same')
        self.maxpool_1 = nn.MaxPool3d(2)
        self.maxpool_2 = nn.MaxPool3d(2)

        self.pos_embed = nn.Embedding(self.bev_num, self.dim)


    def forward(self, x):

        x = self.resnet(x)
        x = layer_norm(x, [x.size()[-4], x.size()[-3], x.size()[-2], x.size()[-1]])
        x = self.bev_map(x)
            
        pos = self.position_embedding(x)
        out = x+pos

        return out
    
    def bev_map(self, x):
        out = rearrange(x, 'b c dy dx dz -> b (dz dx) (dy c)')
        
        return out
    
    def resnet(self, x):
        x_conv3d_1 = self.conv3d_1(x)
        x_maxpool_1 = self.maxpool_1(relu(x_conv3d_1 + self.conv1d_1(x)))

        x_conv3d_2 = self.conv3d_2(x_maxpool_1)
        x_maxpool_2 = self.maxpool_2(relu(x_conv3d_2 + self.conv1d_2(x_maxpool_1)))

        return x_maxpool_2
    
    def position_embedding(self, x):
        #using a learnable 1D-embedding in a raster order
        batch_number, d_bev, dim = x.size()
        pos = torch.arange(d_bev).repeat(batch_number,1).to(device)
        out = self.pos_embed(pos)

        return out