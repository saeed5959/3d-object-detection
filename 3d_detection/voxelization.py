import numpy as np

from core.settings import model_config



def make_voxel(pointcloud_rgb):

    # parameters
    x_bound = model_config.x_bound
    y_bound = model_config.y_bound
    z_bound = model_config.z_bound

    x_voxel_len = model_config.x_voxel_len
    y_voxel_len = model_config.y_voxel_len
    z_voxel_len = model_config.z_voxel_len

    x_voxel_num = int((x_bound[1] - x_bound[0]) / x_voxel_len)
    y_voxel_num = int((y_bound[1] - y_bound[0]) / y_voxel_len)
    z_voxel_num = int((z_bound[1] - z_bound[0]) / z_voxel_len)

    intensity_norm = model_config.intensity_norm

    #remove data out of boundary
    for data in pointcloud_rgb:
        if not ((x_bound[1] > data[0] > x_bound[0]) and (y_bound[1] > data[1] > y_bound[0]) and (z_bound[1] > data[2] > z_bound[0])):
            pointcloud_rgb.remove(data)

    #making a zero grid voxel
    pointcloud_voxel = np.zeros((y_voxel_num, x_voxel_num, z_voxel_num, 5))

    voxel_id = []
    voxel_point = []

    #putting rgb into voxels
    for data in pointcloud_rgb:
        x_id = int((data[0] - x_bound[0]) // x_voxel_len)
        y_id = int((data[1] - y_bound[0]) // y_voxel_len)
        z_id = int((data[2] - z_bound[0]) // z_voxel_len)

        id = (x_id, y_id, z_id)
        if id in voxel_id:
            id_index = voxel_id.index(id)
            voxel_point[id_index].append(data)
        else: 
            voxel_id.append(id)
            voxel_point.append([data]) 

    #convert points in voxel to a 5d vector
    voxel_5d = []
    for voxel in voxel_point:

        intensity, z_mean, r_mean, g_mean, b_mean = 0, 0, 0, 0, 0

        for point in voxel:
            intensity += 1 
            z_mean += point[2]
            r_mean += point[3]
            g_mean += point[4]
            b_mean += point[5]

        z_mean = z_mean / intensity
        r_mean = r_mean / intensity
        g_mean = g_mean / intensity
        b_mean = b_mean / intensity
        intensity = intensity / intensity_norm

        voxel_5d.append([intensity, z_mean, r_mean, g_mean, b_mean])

    #filling grid voxel and for empty voxel the vector will be zero=(0,0,0,0,0)
    for id,voxel in zip(voxel_id, voxel_5d):
        x_id, y_id, z_id = id[0], id[1], id[2]
        print(x_id)
        pointcloud_voxel[y_id, x_id, z_id] = voxel

    return pointcloud_voxel

pointcloud_rgb = [[0.09,0.15,0.16,4,5,6]] #(x,y,z,r,g,b)
pointcloud_voxel = make_voxel(pointcloud_rgb)
print(pointcloud_voxel)