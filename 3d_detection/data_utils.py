import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from einops import rearrange

from core.settings import model_config

class DatasetObjectDetection(Dataset):
    def __init__(self, dataset_file_path):
        super().__init__()
        self.model_config = model_config
        with open(dataset_file_path) as file:
            self.dataset_file = file.readlines()
        
    def get_image(self, data: str, augment: bool):
        #split data with |
        data_list = data.split("|")


        return
    
    def __getitem__(self,index):
        return self.get_image(self.dataset_file[index], augment=False)
    
    def __len__(self):
        return len(self.dataset_file)
    
