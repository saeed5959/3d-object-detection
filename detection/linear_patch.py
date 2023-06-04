import torch
from torch import nn
from torch.nn.functional import layer_norm, relu
from einops import rearrange

from core.settings import model_config, train_config

device = train_config.device

class LinearProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = model_config.input_dim
        self.dim = model_config.dim
        self.bev_num = model_config.bev_num
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(self.dim, 512, kernel_size=3, padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=5, padding='same'),
            nn.BatchNorm2d(512)
        )
        self.conv1d_1 = nn.Conv2d(self.dim, 512, kernel_size=1, padding='same')
        self.maxpool_1 = nn.MaxPool2d(2)

        self.pos_embed = nn.Embedding(self.bev_num, self.dim)

    def forward(self, x):

        x = self.resnet(x)
        x = layer_norm(x, [x.size()[-3], x.size()[-2], x.size()[-1]])
        x = self.divide_patch(x)
                              
        pos = self.position_embedding(x)
        out = x+pos

        return out
    
    def divide_patch(self, x):
        out = rearrange(x, 'b c dx dz -> b (dz dx) c')

        return out

    def resnet(self, x):
        x_conv2d_1 = self.conv2d_1(x)
        x_maxpool_1 = self.maxpool_1(relu(x_conv2d_1 + self.conv1d_1(x)))

        return x_maxpool_1
    
    def position_embedding(self, x):
        #using a learnable 1D-embedding in a raster order
        batch_number, d_bev, dim = x.size()
        pos = torch.arange(d_bev).repeat(batch_number,1).to(device)
        out = self.pos_embed(pos)

        return out