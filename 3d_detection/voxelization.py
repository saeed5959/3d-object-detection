import numpy as np


x_bound = np.array([-10,10])
y_bound = np.array([0,3])
z_bound = np.array([0,20])

x_voxel_len = 0.1
y_voxel_len = 0.1
z_voxel_len = 0.1

x_voxel_num = int((x_bound[1] - x_bound[0]) / x_voxel_len)
y_voxel_num = int((y_bound[1] - y_bound[0]) / y_voxel_len)
z_voxel_num = int((z_bound[1] - z_bound[0]) / z_voxel_len)

pointcloud_rgb = [[1,2,3,4,5,6]] #(x,y,z,r,g,b)

#remove data out of boundary
for data in pointcloud_rgb:
    if not ((x_bound[1] > data[0] > x_bound[0]) and (y_bound[1] > data[1] > y_bound[0]) and (z_bound[1] > data[2] > z_bound[0])):
        pointcloud_rgb.remove(data)

pointcloud_voxel = np.zeros((y_voxel_num, x_voxel_num, z_voxel_num, 6))

voxel_id = []
voxel_point = []

for data in pointcloud_rgb:
    x_id = (data[0] - x_bound[0]) // x_voxel_len
    y_id = (data[1] - y_bound[0]) // y_voxel_len
    z_id = (data[2] - z_bound[0]) // z_voxel_len

    id = int(x_id + y_id*x_voxel_num + z_id*x_voxel_num*y_voxel_num)
    if id in voxel_id:
        id_index = voxel_id.index(id)
        voxel_point[id_index].append(data)
    else: 
        voxel_id.append(id)
        voxel_point.append([data]) 


for voxel in voxel_point:
    for point in voxel:
        intensity = 
        height = 
        density = 


for id,voxel in zip(voxel_id, voxel_point):
    x = id
    y = 
    z = 
    pointcloud_voxel[x,y,z] = voxel