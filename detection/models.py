import torch
from torch import nn
from torch.nn.functional import sigmoid, softmax

from detection import d2_backbone, d3_backbone
from core.settings import model_config


class VoxelTransModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_2d = d2_backbone.Detect2D()
        self.backbone_3d = d3_backbone.Detect3D()

    def forward(self, x):
        d3_out = self.backbone_3d(x)
        d2_out = self.backbone_2d(d3_out)

        return d2_out

