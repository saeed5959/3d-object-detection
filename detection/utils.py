import torch
import cv2
from einops import rearrange


def load_pretrained(model: object, pretrained: str, device: str):

    pretrained_model = torch.load(pretrained, map_location=device)
    model.load_state_dict(pretrained_model)

    return model

