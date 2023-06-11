import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


scan = np.fromfile("/home/saeed/Downloads/MDImagePyramidReference.unknown.0000002.dat", dtype=np.float32)

print(scan[0])