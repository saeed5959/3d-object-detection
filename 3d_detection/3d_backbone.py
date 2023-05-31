import torch
from torch import nn

from core.settings import model_config

class Detect3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = model_config.input_dim
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(self.input_dim, 16, kernel_size=3, padding='same'),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding='same'),
            nn.BatchNorm3d(32),
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

        self.pos_embed = nn.Embedding(self.patch_num,self.dim)


    def forward(self,x):

        return 