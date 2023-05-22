import open3d as o3d

def rgbd_show(file_path: str):
    pcd = o3d.io.read_point_cloud(file_path)

    # print(pcd.points[0])
    o3d.visualization.draw_geometries([pcd])

rgbd_show("./dataset/rgbd/01.ply")