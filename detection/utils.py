import torch
import numpy as np
import cv2
import pypotree
import open3d as o3d
import imageio.v3 as iio

from core.settings import model_config
from detection.image_detection import yolo_detection

def load_pretrained(model: object, pretrained: str, device: str):

    pretrained_model = torch.load(pretrained, map_location=device)
    model.load_state_dict(pretrained_model)

    return model


def make_label(label_path: str):
    label_class = {'Car':1, 'Van':1, 'Truck':1, 'Pedestrian':2, 'Person_sitting':2, 'Cyclist':3, 'Tram':1, 'Misc':4}

    labels = []
    with open(label_path) as file:
        label_file = file.readlines()

    for line in label_file:
        line = line.split(" ")
        if line[0] in label_class:# checking correct class
            cx = ((float(line[4]) + float(line[6])) / 2) / model_config.w_image
            cy = ((float(line[5]) + float(line[7])) / 2) / model_config.h_image
            w = ((float(line[6]) - float(line[4]))) / model_config.w_image
            h = ((float(line[7]) - float(line[5]))) / model_config.h_image
            labels.append({"class":label_class[line[0]], "cx_cy_w_h":(cx,cy,w,h), "H_W_L":(line[8], line[9], line[10]),
                            "tx_ty_tz":(line[11], line[12], line[13]), "rot":line[14], "class_name": line[0]})

    return labels


def calib(calib_path: str):

    with open(calib_path) as file:
       calib_file = file.readlines()

    P_ = [float(m) for m in calib_file[2].split()[1:13]]
    R_ = [float(m) for m in calib_file[4].split()[1:10]]
    t_ = [float(m) for m in calib_file[5].split()[1:13]]

    P = np.array([[P_[0], P_[1], P_[2], P_[3]],
                  [P_[4], P_[5], P_[6], P_[7]],
                  [P_[8], P_[9], P_[10], P_[11]]])
    
    R = np.array([[R_[0], R_[1], R_[2]],
                  [R_[3], R_[4], R_[5]],
                  [R_[6], R_[7], R_[8]]])
    
    t = np.array([[t_[0], t_[1], t_[2], t_[3]],
                  [t_[4], t_[5], t_[6], t_[7]],
                  [t_[8], t_[9], t_[10], t_[11]]])

    return P, R, t


def display_img_label(img_path: str, label_path: str, out_path: str):
    labels = make_label(label_path)

    img = cv2.imread(img_path)
    h, w, c = img.shape
    print(h)
    color = (255,255,255)
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5

    for label in labels:
        box = label["cx_cy_w_h"]
        cv2.rectangle(img, (int((box[0] - box[2]/2)*w), int((box[1] - box[3]/2)*h)), (int((box[0] + box[2]/2)*w), int((box[1] + box[3]/2)*h)), color, thickness)
        cv2.putText(img, label["class_name"], (int(box[0]*w),int(box[1]*h)), font, fontScale, color, thickness, cv2.LINE_AA,)
    cv2.imwrite(out_path,img)

    return


def show_pointcloud_npy(pointcloud_path: str):

    xyz = np.load(pointcloud_path)
    cloudpath = pypotree.generate_cloud_for_display(xyz)
    pypotree.display_cloud_colab(cloudpath)
    
    return


def show_pointcloud_bin(pointcloud_path: str):

    scan = np.fromfile(pointcloud_path, dtype=np.float32)

    xyz_list = []
    for i in range(len(scan)):
        if i%4 == 0 :
            x, y, z = scan[i], scan[i+1], scan[i+2]
            xyz_list.append([x, y, z])

    xyz = np.array(xyz_list)
    cloudpath = pypotree.generate_cloud_for_display(xyz)
    pypotree.display_cloud_colab(cloudpath)
    
    return


def show_pointcloud_npy_open3d(pointcloud_path: str):

    xyz = np.load(pointcloud_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

    return


def show_pointcloud_npy_open3d_color(pointcloud_path: str):

    xyz = np.load(pointcloud_path)
    color = np.load(pointcloud_path[:-4]+"_color.npy")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(color/256)
    o3d.visualization.draw_geometries([pcd])

    return


def display_pointcloud_label():

    return



def display_pointcloud_mask(pointcloud_path: str, label_path: str, img_path: str, calib_path: str, out_path: str):
    labels = make_label(label_path)
    img = iio.imread(img_path)

    pointclouds = np.fromfile(pointcloud_path, dtype=np.float32)
    
    P, R, t = calib(calib_path)

    x_bound = model_config.x_bound
    y_bound = model_config.y_bound
    z_bound = model_config.z_bound

    # in camera coordinate : angle in x-axis  = (-x_theta,x_theta)
    x_angle = np.tan(np.pi/180*model_config.x_theta)
    # y_angle = (h/w) * x_angle
    y_angle = np.tan(np.pi/180*model_config.y_theta)

    pointclouds_transform = []
    color = []
    for i in range(len(pointclouds)):
        if i%4 == 0 :

            # transformation from lidar coordinate to camera coordinate 
            point = np.array([pointclouds[i], pointclouds[i+1], pointclouds[i+2], 1])
            point_transform = np.matmul(R, np.matmul(t, point))
            x, y, z = point_transform[0], point_transform[1], point_transform[2]

            if (x_bound[1] > x > x_bound[0]) and (y_bound[1] > y > y_bound[0]) and (z_bound[1] > z > z_bound[0]):# and (np.abs(x) < x_angle*z) and (np.abs(y) < y_angle*z):
                

                point = np.array([x, y, z, 1])
                point_transform = np.matmul(P, point)
                point_transform = point_transform / point_transform[2] # divide to z

                if (model_config.h_image > int(point_transform[1]) >= 0) and (model_config.w_image > int(point_transform[0]) >= 0):
                    color.append(img[int(point_transform[1]), int(point_transform[0]), : ])
                    pointclouds_transform.append([x,y,z])

    np.save(out_path, np.array(pointclouds_transform))
    np.save(out_path[:-4]+"_color.npy", np.array(color))
    
    obj_class, obj_boxes_norm, masks = yolo_detection(img_path)
    kernel = np.ones((9,9),np.uint8)
    
    # for count, label in enumerate(obj_class):
        
    #     pointclouds_label = []
    #     cx, cy, w, h = obj_boxes_norm[count]
    #     x_angle_max = np.tan(np.pi/180*model_config.x_theta) * (cx + w/2 - 0.5) / 0.5
    #     x_angle_min = np.tan(np.pi/180*model_config.x_theta) * (cx - w/2 - 0.5) / 0.5
    #     y_angle_max = np.tan(np.pi/180*model_config.y_theta) * (cy + h/2 - 0.5) / 0.5
    #     y_angle_min = np.tan(np.pi/180*model_config.y_theta) * (cy - h/2 - 0.5) / 0.5

    #     for point in pointclouds_transform:
    #         x, y, z = point[0], point[1], point[2] 
            
    #         if (x_angle_max*z > x > x_angle_min*z) and (y_angle_max*z > y > y_angle_min*z):
    #             x_img = int((x_angle*z + x) / (2*x_angle*z) * model_config.w_image)
    #             y_img = int((y_angle*z + y) / (2*y_angle*z) * model_config.h_image)

    #             if masks[count].data[0][y_img, x_img]==1 :
    #                 pointclouds_label.append([x,y,z])


    for count, mask in enumerate(masks):

        pointclouds_label = []
        color_label = []
        mask_erode = cv2.erode(mask.data[0], kernel,iterations = 3)
        
        for p in pointclouds_transform:
            # transformation from camera coordinate to image coordinate 
            point = np.array([p[0], p[1], p[2], 1])
            point_transform = np.matmul(P, point)
            point_transform = point_transform / point_transform[2] # divide to z
            
            
            if mask_erode[int(point_transform[1]), int(point_transform[0])]==1 :
                pointclouds_label.append(p)
                color_label.append(img[int(point_transform[1]), int(point_transform[0]), : ])

        out_label_path = out_path[:-4] + f"_{count}.npy"
        np.save(out_label_path, np.array(pointclouds_label))
        np.save(out_label_path[:-4]+"_color.npy", np.array(color_label))

    return 