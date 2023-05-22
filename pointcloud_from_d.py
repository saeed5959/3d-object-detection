import numpy as np
import open3d as o3d
import cv2


def ptc_from_rgbd(depth_path: str, img_path:str):
    depth_img = cv2.imread(depth_path,0)
    img = cv2.imread(img_path)
    #changing channel 0 and 2 for o3d color
    img_0 = np.copy(img[:,:,0])
    img[:,:,0] = img[:,:,2]
    img[:,:,2] = img_0

    # get depth image resolution:
    height, width = depth_img.shape
    # compute indices:
    jj = np.tile(range(width), height)
    ii = np.repeat(range(height), width)
    # Compute constants:
    xx = (jj - CX_DEPTH) / FX_DEPTH
    yy = (ii - CY_DEPTH) / FY_DEPTH
    # transform depth image to vector of z:
    length = height * width
    z = depth_img.reshape(length)
    # compute point cloud
    pcd = np.dstack((xx * z, yy * z, z)).reshape((length, 3))

    # cam_RGB = np.apply_along_axis(np.linalg.inv(R).dot, 1, pcd) - np.linalg.inv(R).dot(T)
    # xx_rgb = ((cam_RGB[:, 0] * FX_DEPTH) / cam_RGB[:, 2] + CX_DEPTH + width / 2).astype(int).clip(0, width - 1)
    # yy_rgb = ((cam_RGB[:, 1] * FY_DEPTH) / cam_RGB[:, 2] + CY_DEPTH).astype(int).clip(0, height - 1)
    # colors = img[yy_rgb, xx_rgb]
    colors = img.reshape((length,3))

    return pcd, colors

def show_ptc(pcd, colors):
    # Convert to Open3D.PointCLoud:
    pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
    pcd_o3d.colors = o3d.utility.Vector3dVector(np.array(colors / 255))
    # Visualize:
    o3d.visualization.draw_geometries([pcd_o3d])


if __name__=="__main__":
    FX_DEPTH = 574.334300
    FY_DEPTH = 574.334300
    CX_DEPTH = 320
    CY_DEPTH = 240

    R = np.array([[0.999999, 0.001190, 0.000299],
                [-0.001190, 0.881206, 0.472731],
                [0.000299, -0.472731, 0.881207]])

    T = np.array([0.0, 0.0, 0.0])

    # Read depth image:
    depth_path = "./dataset/rgbd/1/depth/2015-11-08T13.42.16.610-0000006334.png"
    img_path = "./dataset/rgbd/1/image/2015-11-08T13.42.16.610-0000006334.jpg"

    ptc, colors = ptc_from_rgbd(depth_path, img_path)
    show_ptc(ptc, colors)