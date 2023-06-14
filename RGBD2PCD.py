import argparse
import sys
import os
import numpy as np
import open3d as o3d
from PIL import Image

# Pyrealsense
depth_scaling_factor = 999.99
f_x = 618.212
f_y=617.673
c_x = 323.716
c_y = 256.977
k1=0
k2=0


for i in range(1,21):
    depth=Image.open("./zigzag/data/align_test_depth{}.png".format(i))
    rgb=Image.open("./zigzag/data/align_test{}.png".format(i))


    colors = []
    points = []
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color = rgb.getpixel((u, v))
            Z = depth.getpixel((u, v)) / depth_scaling_factor
            X = (u - c_x) * Z / f_x
            Y = (v - c_y) * Z / f_y
            if Z>1.0:
                continue
            points.append([X,Y,Z])
            colors.append([color[0]/255.0,color[1]/255.0,color[2]/255.0])

    points=np.array(points)
    colors=np.array(colors)

    # Convert to Open3D.PointCLoud:
    pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(points)
    pcd_o3d.colors=o3d.utility.Vector3dVector(colors)

    # Visualize:
    o3d.io.write_point_cloud("./zigzag/pcd/result{}.pcd".format(i),pcd_o3d)
