import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def matplot_show(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='.',)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')

    plt.show()
    return


def open3d_show(xyz, voxel=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.colors = o3d.utility.Vector3dVector(colors/65535)

    if voxel:
        print("voxel activated")
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.1)
        o3d.visualization.draw_geometries([voxel_grid])
    else:
        o3d.visualization.draw_geometries([pcd])

    return


scan = np.fromfile("./dataset/lidar/0000000000.bin", dtype=np.float32)

x_list = []
y_list = []
z_list = []
xyz_list = []

for i in range(len(scan)):
    if i%4 == 0:
        if scan[i+1]<1000:
            x_list.append(scan[i])
            y_list.append(scan[i+1])
            z_list.append(scan[i+2])
        xyz_list.append([scan[i],scan[i+1],scan[i+2]])

x = np.array(x_list)
y = np.array(y_list)
z = np.array(z_list)
xyz = np.array(xyz_list)

open3d_show(xyz, voxel=True)
#matplot_show(x,y,z)