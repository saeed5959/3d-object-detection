import json
import os
import cv2
import numpy as np
from tqdm import tqdm

from detection.voxelization import make_voxel
from core.settings import model_config

def save_file(file_path: str, data_file: list):

    with open(file_path, "w") as file:
        file.write(data_file[0]) 
        for line in data_file[1:]:
            file.write("\n" + line)

    return


def alignment(rgb, depth, pointcloud):
    pointcloud_rgb = []

    return pointcloud_rgb


def make_alignment_voxel(rgb_path: str, d_path: str, pointcloud_path: str, out_path: str):

    rgb = cv2.imread(rgb_path)

    depth = cv2.imread(d_path,0)

    pointcloud = np.fromfile(pointcloud_path, dtype=np.float32)

    #alignment between rgbd and pointcloud
    pointcloud_rgb = alignment(rgb, depth, pointcloud)

    #voxelization
    pointcloud_rgb_voxel = make_voxel(pointcloud_rgb)

    with open(out_path, 'wb') as f:
        np.save(f, pointcloud_rgb_voxel)

    return


def main(rgb_folder: str, d_folder: str, pointcloud_folder: str, out_folder: str, fil_path_out: str):

    dataset = []
    for rgb_name in os.listdir(rgb_folder):
        d_name = rgb_name[:-3] + "png"
        pointcloud_name = rgb_name[:-3] + "bin"
        out_name = rgb_name[:-3] + "bin"

        rgb_path = os.path.join(rgb_folder, rgb_name)
        d_path = os.path.join(d_folder, d_name)
        pointcloud_path = os.path.join(pointcloud_folder, pointcloud_name)
        out_path = os.path.join(out_folder, out_name)

        make_alignment_voxel(rgb_path, d_path, pointcloud_path, out_path)
        dataset.append(out_path)

    save_file(file_path_out, dataset)

    return 


rgb_folder = ""
d_folder = ""
pointcloud_folder = ""
out_folder = ""
file_path_out = "./dataset/dataset_out.txt"

main(rgb_folder, d_folder, pointcloud_folder, out_folder, file_path_out)
