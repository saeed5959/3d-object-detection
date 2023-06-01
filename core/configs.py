"""
    All config is in here
"""
import torch

class ModelConfig:
    """
        All model config
    """

    def __init__(self):
        self.x_bound: list = [-0.2, 0.2]#[-8, 8]
        self.y_bound: list =[0,0.2]#[0, 4]
        self.z_bound: list = [-0.2,0.2]#[-16, 16]
        self.x_voxel_len: int = 0.1
        self.y_voxel_len: int = 0.1
        self.z_voxel_len: int = 0.1
        self.intensity_norm: int = 10 #1000

        self.input_dim: int = 5
        self.dim: int = int(64*self.y_bound[1]/self.y_voxel_len/4)#640
        self.bev_num: int = int(((self.x_bound[1]-self.x_bound[0]) / self.x_voxel_len / 4) * ((self.z_bound[1]-self.z_bound[0]) / self.z_voxel_len / 4))#3200
        self.head_num: int = 2
        self.class_num: int = 10
                                                                         


class TrainConfig:
    """
        All train config
    """

    def __init__(self):
        self.save_model: int = 100
        self.epochs: int = 100
        self.batch_size: int = 48
        self.learning_rate: float = 0.0001
        self.step_show: int = 100
        self.device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
