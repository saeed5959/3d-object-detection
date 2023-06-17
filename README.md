# 3D-Object-Detection
## 3D Object Detection by Colorful Pointcloud
## Making an alignment between (Lidar pointcloud) and (RGB image)



### Model Architecture
<img src="/3d_detection.png" width="900" height="300" border="20" title="model">

---

### Alignment
<img src="/alignment.png" width="900" height="300" border="20" title="model">

---

## Color of Pointcloud Matters!
### 3D Object Detection by Colorful Pointcloud
### Alignment Between RGB Pointcloud and Lidar Pointcloud

Abstract
One of the most critical task in autonomous vehicle is to detect different kind of object in
the environment. Detecting these objects are important due to different reaction we have
with them. Finding these objects without the colors or fusing lidar pointcloud with image ,
without aligning color of pixels of image to the lidar pointcloud, is hard for the model to
understand it. But in this work, we first align image and pointcloud lidar and then give a
color to the every poitcloud, and then with these colorful pointcloud we can detect
different objects with a voxel-based CNN-Transformer model that has a better
performance to the other models.