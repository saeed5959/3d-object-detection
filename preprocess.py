import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import imageio.v3 as iio

from detection.voxelization import make_voxel
from core.settings import model_config


def save_file(file_path: str, data_file: list):

    with open(file_path, "w") as file:
        file.write(data_file[0]) 
        for line in data_file[1:]:
            file.write("\n" + line)

    return


def alignment(rgb_path, pointcloud_path):

    pointcloud_list = np.fromfile(pointcloud_path, dtype=np.float32)
    #img = cv2.imread(img_path)
    # o3d accept order in rgb but cv2 order is bgr 
    img = iio.imread(rgb_path)
    h, w, _ = img.shape

    pointcloud = []
    color = []
    pointcloud_color = []

    R = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04], 
                [1.480249e-02, 7.280733e-04, -9.998902e-01],
                [9.998621e-01, 7.523790e-03, 1.480755e-02]])

    t = np.array([-4.069766e-03, -7.631618e-02,-2.717806e-01])

    #y-axis : angle_y=41 >> tg(angle_y) = (w/2)/f = y/x >>  w/2 = 0.81*f , y==0.81*x 
    #z-axis : angle=?  , tg(angle_z) = (h/2)/f = (h/2)/(w/(0.81*2)) = z/x >>  z==((h/w)*0.81)*x     >> angle_z = 15
    y_agnle = np.tan(np.pi/180*41)
    z_angle = (h/w) * y_agnle

    #scan have 4 value : x,y,z,0
    for i in range(len(pointcloud_list)):
        if i%4 == 0 :

            # transformation from lidar coordinate to camera coordinate 
            point = np.array([pointcloud_list[i], pointcloud_list[i+1], pointcloud_list[i+2]])
            point_transform = np.matmul(R, np.transpose(point)) + np.transpose(t)
            y, z, x = -1*point_transform[0], -1*point_transform[1], point_transform[2]

            if (x<80 and x>0) and (y<40 and y>-40) and (z<4 and z>-4) and (np.abs(y) < y_agnle*x) and (np.abs(z) < z_angle*x):
                #pointcloud
                pointcloud.append([x, y, z])

                #color
                x_img = (y_agnle*x-y) / (2*y_agnle*x) * w
                y_img = (z_angle*x - z) / (2*z_angle*x) * h
                rgb = img[int(y_img), int(x_img), :]
                color.append(rgb)

                #pointcloud + color
                r, g, b = rgb[0], rgb[1], rgb[2]
                pointcloud_color.append([x, y, z, r, g, b])

                

    pointcloud = np.array(pointcloud)
    color = np.array(color)
    pointcloud_color = np.array(pointcloud_color)

    return pointcloud_color


def make_alignment_voxel(rgb_path: str, pointcloud_path: str, out_path: str):

    #alignment between rgbd and pointcloud
    pointcloud_rgb = alignment(rgb_path, pointcloud_path)

    #voxelization
    pointcloud_rgb_voxel = make_voxel(pointcloud_rgb)

    with open(out_path, 'wb') as f:
        np.save(f, pointcloud_rgb_voxel)

    return


def main(rgb_folder: str, pointcloud_folder: str, out_folder: str, file_path_out: str):

    dataset = []
    for rgb_name in os.listdir(rgb_folder):
        pointcloud_name = rgb_name[:-3] + "bin"
        out_name = rgb_name[:-3] + "bin"

        rgb_path = os.path.join(rgb_folder, rgb_name)
        pointcloud_path = os.path.join(pointcloud_folder, pointcloud_name)
        out_path = os.path.join(out_folder, out_name)

        make_alignment_voxel(rgb_path, pointcloud_path, out_path)
        dataset.append(out_path)

    save_file(file_path_out, dataset)

    return 


rgb_folder = ""
pointcloud_folder = ""
out_folder = ""
file_path_out = "./dataset/dataset_out.txt"

main(rgb_folder, pointcloud_folder, out_folder, file_path_out)
