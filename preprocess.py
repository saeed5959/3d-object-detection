import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import imageio.v3 as iio

from core.settings import model_config
from detection.image_detection import yolo_detection
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


def make_voxel(pointcloud_path):

    # in lidar coordinate 
    pointclouds = np.fromfile(pointcloud_path, dtype=np.float32)

    R = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04], 
                [1.480249e-02, 7.280733e-04, -9.998902e-01],
                [9.998621e-01, 7.523790e-03, 1.480755e-02]])

    t = np.array([-4.069766e-03, -7.631618e-02,-2.717806e-01])
    
    # in camera coordinate : angle in x-axis  = (-x_theta,x_theta)
    x_angle = np.tan(np.pi/180*model_config.x_theta)
    # y_angle = (h/w) * x_angle
    y_angle = np.tan(np.pi/180*model_config.y_theta)

    x_bound = model_config.x_bound
    y_bound = model_config.y_bound
    z_bound = model_config.z_bound

    x_voxel_len = model_config.x_voxel_len
    y_voxel_len = model_config.y_voxel_len
    z_voxel_len = model_config.z_voxel_len

    x_voxel_num = int((x_bound[1] - x_bound[0]) / x_voxel_len)
    y_voxel_num = int((y_bound[1] - y_bound[0]) / y_voxel_len)
    z_voxel_num = int((z_bound[1] - z_bound[0]) / z_voxel_len)

    voxel = np.zeros((x_voxel_num, y_voxel_num, z_voxel_num))

    #scan have 4 value : x,y,z,0
    for i in range(len(pointclouds)):
        if i%4 == 0 :

            # transformation from lidar coordinate to camera coordinate 
            point = np.array([pointclouds[i], pointclouds[i+1], pointclouds[i+2]])
            point_transform = np.matmul(R, np.transpose(point)) + np.transpose(t)
            x, y, z = point_transform[0], point_transform[1], point_transform[2]

            if (x_bound[1] > x > x_bound[0]) and (y_bound[1] > y > y_bound[0]) and (z_bound[1] > z > z_bound[0]) and (np.abs(x) < x_angle*z) and (np.abs(y) < y_angle*z):

                x_id = int((x - x_bound[0]) // x_voxel_len)
                y_id = int((y - y_bound[0]) // y_voxel_len)
                z_id = int((z - z_bound[0]) // z_voxel_len)
                voxel[y_id, x_id, z_id] += 1

    return voxel


def make_mask(obj_box_norm, voxel_shape):

    cx, cy, w, h = obj_box_norm[0], obj_box_norm[1], obj_box_norm[3], obj_box_norm[4]
    x_angle_max = np.tan(np.pi/180*model_config.x_theta) * (cx + w/2 - 0.5) / 0.5
    x_angle_min = np.tan(np.pi/180*model_config.x_theta) * (cx - w/2 - 0.5) / 0.5
    y_angle_max = np.tan(np.pi/180*model_config.y_theta) * (cy + h/2 - 0.5) / 0.5
    y_angle_min = np.tan(np.pi/180*model_config.y_theta) * (cy - h/2 - 0.5) / 0.5

    mask = np.zeros(voxel_shape)

    x_bound = model_config.x_bound
    y_bound = model_config.y_bound
    z_bound = model_config.z_bound

    x_voxel_len = model_config.x_voxel_len
    y_voxel_len = model_config.y_voxel_len
    z_voxel_len = model_config.z_voxel_len

    x_voxel_num = int((x_bound[1] - x_bound[0]) / x_voxel_len)
    y_voxel_num = int((y_bound[1] - y_bound[0]) / y_voxel_len)
    z_voxel_num = int((z_bound[1] - z_bound[0]) / z_voxel_len)

    for x in range(x_voxel_num):
        for y in range(y_voxel_num):
            for z in range(z_voxel_num):
                if (x_angle_max*z > x > x_angle_min*z) and (y_angle_max*z > y > y_angle_min*z):
                    mask[x, y, z] = 1

    return mask


def make_3d_target(obj_box_norm, label_file):


    return coordinate_target_3d


def main(rgb_folder: str, pointcloud_folder: str, out_folder: str, file_path_out: str):    

    intensity_max = 0
    dataset = []
    for rgb_name in os.listdir(rgb_folder):
        pointcloud_name = rgb_name[:-3] + "bin"
        out_name = rgb_name[:-3] + "bin"

        rgb_path = os.path.join(rgb_folder, rgb_name)
        pointcloud_path = os.path.join(pointcloud_folder, pointcloud_name)
        out_path = os.path.join(out_folder, out_name)

        objs_class, objs_box_norm = yolo_detection(rgb_path)
        voxel = make_voxel(pointcloud_path)
        if np.max(voxel) > intensity_max:
            intensity_max = np.max(voxel)

        for obj_class, obj_box_norm in zip(objs_class, objs_box_norm):
            coordinate_target_3d = make_3d_target(obj_box_norm, label_file)
            if coordinate_target_3d != []:
                mask = make_mask(obj_box_norm, voxel.shape)
                voxel_mask = voxel * mask
                np.save(out_path, voxel_mask)
                dataset.append(f"{out_path}|{obj_class}|{coordinate_target_3d}")

    save_file(file_path_out, dataset)

    return 


rgb_folder = "./dataset/rgb"
pointcloud_folder = "./dataset/pointcloud"
out_folder = "./dataset/out"
file_path_out = "./dataset/dataset_out.txt"
label_file = ""

main(rgb_folder, pointcloud_folder, out_folder, file_path_out, label_file)
