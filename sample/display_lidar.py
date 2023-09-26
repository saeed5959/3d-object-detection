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


def graphic_view(xyz):
    import pypotree
    cloudpath = pypotree.generate_cloud_for_display(xyz)
    pypotree.display_cloud_colab(cloudpath)


scan = np.fromfile("/home/saeed/software/python/pointcloud/git_code/sample/kitty/0000000024.bin", dtype=np.float32)
# scan = np.fromfile("/home/saeed/software/python/pointcloud/git_code/sample/kitty/0000000024.bin", dtype=np.float32)

x_list = []
y_list = []
z_list = []
xyz_list = []

#scan have 4 value : x,y,z,0
for i in range(len(scan)):
    if i%4 == 0 :
        y, z, x = scan[i], scan[i+1], scan[i+2]
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
        xyz_list.append([x, y, z])

x = np.array(x_list)
y = np.array(y_list)
z = np.array(z_list)
xyz = np.array(xyz_list)

# print(scan[0:10])
# open3d_show(xyz, voxel=False)
# matplot_show(x,y,z)
graphic_view(xyz)