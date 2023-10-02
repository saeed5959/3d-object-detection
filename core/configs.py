"""
    All config is in here
"""
import torch
import numpy as np

class ModelConfig:
    """
        All model config
    """

    def __init__(self):
        self.path_yolo: str = "./dataset/pretrain_yolo/yolov8x-seg.pt"
        self.x_bound: list = [-30, 30]
        self.y_bound: list =[-3, 3]
        self.z_bound: list = [0, 60]
        self.x_voxel_len: int = 0.05
        self.y_voxel_len: int = 0.1
        self.z_voxel_len: int = 0.05
        self.h_image: int = 376
        self.w_image: int = 1241
        self.x_theta: int = 41
        self.y_theta: int = np.arctan((self.h_image/self.w_image)*np.tan(np.pi/180*self.x_theta))*180/np.pi
        self.intensity_norm: int = 10 #1000
        self.downsample: int = 8#2^num_max_pooling

        self.input_dim: int = 5
        self.dim: int = int(64 * self.y_bound[1] / self.y_voxel_len / self.downsample)#320
        self.bev_num: int = int(((self.x_bound[1]-self.x_bound[0]) / self.x_voxel_len / self.downsample) * ((self.z_bound[1]-self.z_bound[0]) / self.z_voxel_len / self.downsample))#3200
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
