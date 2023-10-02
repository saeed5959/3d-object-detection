import os
import numpy as np

from core.settings import model_config
from detection.utils import make_label


def save_file(file_path: str, data_file: list):

    with open(file_path, "w") as file:
        file.write(data_file[0]) 
        for line in data_file[1:]:
            file.write("\n" + line)

    return


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


def make_mask(label, voxel_shape):

    cx, cy, w, h = label["cx_cy_w_h"]
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


def main(pointcloud_folder: str, out_folder: str, file_path_out: str, label_folder: str):    

    intensity_max = 0
    dataset = []
    for point_name in os.listdir(pointcloud_folder):
        label_name = point_name[:3] + "txt"
        out_name = point_name[:-3] + "bin"

        pointcloud_path = os.path.join(pointcloud_folder, point_name)
        label_path = os.path.join(label_folder, label_name)
        out_path = os.path.join(out_folder, out_name)

        labels = make_label(label_path)
        voxel = make_voxel(pointcloud_path)
        if np.max(voxel) > intensity_max:
            intensity_max = np.max(voxel)

        for label in labels:
            mask = make_mask(label, voxel.shape)
            voxel_mask = voxel * mask
            np.save(out_path, voxel_mask)
            dataset.append(f"{out_path}|{label['class']}|{label['H_W_L']}|{label['tx_ty_tz']|{label['rot']}}")
            # remember to normalize tx_ty_tz and rot in data loader
    save_file(file_path_out, dataset)

    with open("./intensity_max.txt", "w") as file:
        file.write(str(intensity_max)) 

    return 

img_folder = "./dataset/img"
pointcloud_folder = "./dataset/pointcloud"
out_folder = "./dataset/out"
file_path_out = "./dataset/dataset_out.txt"
label_folder = "./dataset/label"

main(pointcloud_folder, out_folder, file_path_out, label_folder)
