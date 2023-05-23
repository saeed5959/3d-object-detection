# pointcloud-align-rgbd
## Making an alignment between (Lidar pointcloud) and (RGBD pointcloud)

---

Problems

    1-lidar pointcloud is accurate but without color, but rgbd pointcloud is not accurate but with color
    2-detecting object in lidar pointcloud is hard but if you found it, the coordinate will be precise
    3-detecting object in rgbd is easy but if you found it, the coordinatre will not be precise

Solution

    1-putting lidar pointcloud as a base pointcloud, because the coordiante of object is important
    2-making a alignment between rgbd and lidar pointcloud
    3-putting the color of rgbd pointcloud into lidar pointcloud

Approaches

    1-getting pointcloud of the whole environment by rgbd and lidar pointcloud seperately and then making alignment
    2-getting pointcloud of the a specific angel of envirenment by rgbd and lidar pointcloud in a same time by fixing lidar and rgbd sensor in a same position and then making a alignment 

---

### Model Architecture
<img src="/3d_detection.png" width="900" height="300" border="20" title="model">