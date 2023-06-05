import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange

from core.settings import model_config

class DatasetObjectDetection(Dataset):
    def __init__(self, dataset_file_path):
        super().__init__()
        self.model_config = model_config
        with open(dataset_file_path) as file:
            self.dataset_file = file.readlines()
        
    def get_pointcloud_voxel(self, data_path: str):
        
        pointcloud_voxel = np.fromfile(data_path, dtype=np.float32)

        #change voxelize input dimention from (y,x,z,5) to (5,y,x,z)
        pointcloud_voxel = rearrange(pointcloud_voxel, 'x y z c -> c x y z')

        pointcloud_voxel = torch.Tensor(pointcloud_voxel)

        return pointcloud_voxel
    
    def __getitem__(self,index):
        return self.get_pointcloud_voxel(self.dataset_file[index])
    
    def __len__(self):
        return len(self.dataset_file)
    
