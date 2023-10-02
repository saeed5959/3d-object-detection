from detection.image_detection import yolo_detection
from detection.utils import display_img_label, display_pointcloud_mask, show_pointcloud_npy, show_pointcloud_npy_open3d, calib, show_pointcloud_npy_open3d_color

def img_show():
    obj_class, obj_boxes_norm, masks = yolo_detection("./sample/kitty/005935.png")
    print(masks)
    display_img_label("./sample/kitty/003095.png", "./sample/label_2/003095.txt", "./sample/out/003095.png")

    return

def pointcloud_mask():

    display_pointcloud_mask("./sample/kitty/005935.bin", "./sample/label_2/005935.txt", "./sample/kitty/005935.png", "./sample/calib/005935.txt", "./sample/out/005935_new/005935.npy")

    return

def pointcloud_show():
    
    show_pointcloud_npy_open3d("./sample/out/005935_new/005935_0.npy")

    return

def pointcloud_show_color():

    show_pointcloud_npy_open3d_color("./sample/out/005935_new/005935_1.npy")

    return

pointcloud_show_color()
