import numpy as np
import open3d as o3d
import cv2
import imageio.v3 as iio

def open3d_show(xyz, color, voxel=False, color_show=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color_show:
        pcd.colors = o3d.utility.Vector3dVector(color/256)

    if voxel:
        print("voxel activated")
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.1)
        o3d.visualization.draw_geometries([voxel_grid])
    else:
        o3d.visualization.draw_geometries([pcd])

    return


def point_color(lidar_path: str, img_path: str):

    pointcloud_list = np.fromfile(lidar_path, dtype=np.float32)
    #img = cv2.imread(img_path)
    # o3d accept order in rgb but cv2 order is bgr 
    img = iio.imread(img_path)
    h, w, _ = img.shape

    pointcloud = []
    color = []

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
                pointcloud.append([x, y, z])

                #color
                x_img = (y_agnle*x-y) / (2*y_agnle*x) * w
                y_img = (z_angle*x - z) / (2*z_angle*x) * h
                color.append(img[int(y_img), int(x_img), :])

                

    pointcloud = np.array(pointcloud)
    color = np.array(color)
    open3d_show(pointcloud, color, voxel=False, color_show=True)




if __name__ == "__main__":

    lidar_path = "/home/saeed/software/python/pointcloud/git_code/sample/kitty/0000000000.bin"
    img_path = "/home/saeed/software/python/pointcloud/git_code/sample/kitty/0000000000.png"

    point_color(lidar_path, img_path)