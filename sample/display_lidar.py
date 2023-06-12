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


scan = np.fromfile("/home/saeed/software/python/pointcloud/git_code/sample/kitty/0000000000.bin", dtype=np.float32)
# scan = np.fromfile("/home/saeed/software/python/pointcloud/git_code/sample/kitty/0000000024.bin", dtype=np.float32)

x_list = []
y_list = []
z_list = []
xyz_list = []

R = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04], 
              [1.480249e-02, 7.280733e-04, -9.998902e-01],
              [9.998621e-01, 7.523790e-03, 1.480755e-02]])

t = np.array([-4.069766e-03, -7.631618e-02,-2.717806e-01])

#scan have 4 value : x,y,z,0
for i in range(len(scan)):
    if i%4 == 0 :
        point = np.array([scan[i], scan[i+1], scan[i+2]])
        point_transform = np.matmul(R, np.transpose(point)) + np.transpose(t)
        y, z, x = -1*point_transform[0], -1*point_transform[1], point_transform[2]
        # print(point)
        # print(x, y, z)
        if (x<80 and x>0) and (y<40 and y>-40) and (z<4 and z>-4) and (x>np.abs(float(y))):
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
            xyz_list.append([x, y, z])

x = np.array(x_list)
y = np.array(y_list)
z = np.array(z_list)
xyz = np.array(xyz_list)

# print(scan[0:10])
open3d_show(xyz, voxel=False)
#matplot_show(x,y,z)