import torch
from torch import nn

from detection import transformer, head

class Detect2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_block = transformer.Transformer()
        self.head_block = head.HeadDetect()

    def forward(self, x):
        transformer_out = self.transformer_block(x)
        out = self.head_block(transformer_out)

        return out
    

