import copy
import numpy as np
import open3d as o3d
import cv2
from scipy.optimize import least_squares
from SIFT import *
import matplotlib.pyplot as plt
import time
import random
from scipy.spatial.transform import Rotation as R
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Bundle Adjustment')
parser.add_argument('--dataset', default="custard")
args = parser.parse_args()

if __name__ == "__main__":

    if args.dataset=='custard':
        # Set image path
        img_path=['./dataset/data/align_test{}.png'.format(i) for i in [4,1,2,3]]
        depth_path=['./dataset/data/align_test_depth{}.png'.format(i) for i in [4,1,2,3]]
        pcd=[o3d.io.read_point_cloud('./dataset/pcd/result{}.pcd'.format(i)) for i in [4,1,2,3]]
        distance_ratio=0.6

    elif args.dataset=='trans':
        # Set image path
        img_path=['./dataset/vatican/data/align_test{}.png'.format(i) for i in range(1,19)]
        depth_path=['./dataset/vatican/data/align_test_depth{}.png'.format(i) for i in range(1,19)]
        pcd=[o3d.io.read_point_cloud('./dataset/vatican/pcd/result{}.pcd'.format(i)) for i in range(1,19)]
        distance_ratio=0.2

    elif args.dataset=='rotation':
        # Set image path
        img_path=['./dataset/desk2/data/align_test{}.png'.format(i) for i in range(1,14)]
        depth_path=['./dataset/desk2/data/align_test_depth{}.png'.format(i) for i in range(1,14)]
        pcd=[o3d.io.read_point_cloud('./dataset/desk2/pcd/result{}.pcd'.format(i)) for i in range(1,14)]
        distance_ratio=0.2

    elif args.dataset=='round':
        # Set image path
        img_path=['./dataset/round/data/align_test{}.png'.format(i) for i in range(1,17)]
        depth_path=['./dataset/round/data/align_test_depth{}.png'.format(i) for i in range(1,17)]
        pcd=[o3d.io.read_point_cloud('./dataset/round/pcd/result{}.pcd'.format(i)) for i in range(1,17)]
        distance_ratio=0.2

    elif args.dataset=='zigzag':
        # Set image path
        img_path=['./dataset/zigzag/data/align_test{}.png'.format(i) for i in range(1,21)]
        depth_path=['./dataset/zigzag/data/align_test_depth{}.png'.format(i) for i in range(1,21)]
        pcd=[o3d.io.read_point_cloud('./dataset/zigzag/pcd/result{}.pcd'.format(i)) for i in range(1,21)]
        distance_ratio=0.2
    ########################################################################################################################
    # Feature matching using SIFT algorithm
    ########################################################################################################################
    # Find transformation matrix from corresponding points based on SIFT
    # Read image from path

    image=[cv2.imread(path) for path in img_path]
    depth=[np.array(o3d.io.read_image(path), np.float32) for path in depth_path]

    # Find keypoints and descriptors using SIFT
    sift=cv2.SIFT_create(nOctaveLayers=5)

    sift_result=[(sift.detectAndCompute(img,None)) for img in image]
    boundary=[(get_boundary(p)) for p in pcd]


    # Estimate Relative Pose
    print("Pose Estimation")
    relative_poses=[]
    for i in range(len(img_path)-1):
        relative_pose, _ = Pose_Estimation(
            sift_result[i], sift_result[i + 1], img_path[i], img_path[i + 1], depth_path[i], depth_path[i + 1], pcd[i],
            pcd[i + 1], distance_ratio=distance_ratio, samples=4, tol=1e-2)

        relative_poses.append(relative_pose)
    print("Pose Estimation Done")

    # Get Covisibility Graph
    print("Covisibility Graph")
    covisibility, observations = get_covisibility(sift_result, img_path, depth_path, pcd,
                                                          distance_ratio=distance_ratio, tol=1e-2)
    print("Covisibility Done")

    # Get Global Poses
    relative_pose_forward=relative_poses
    """1->2, 2->3, 3->4 ...."""

    forward_global_pose = get_global_poses(relative_pose_forward)
    """1->1, 1->2, 1->3, 1->4"""

    inverse_global_pose=[np.linalg.inv(forward_global_pose[i]) for i in range(len(forward_global_pose))]
    """1<-1, 1<-2, 1<-3, 1<-4"""

    results=list()
    for i in range(len(inverse_global_pose)):
        c=copy.deepcopy(pcd[i])
        results.append(c.transform(inverse_global_pose[i]))

    result=results[0]
    for i in range(1,len(results)):
        result+=results[i]
    o3d.visualization.draw_geometries([result])

    # Get 3D coordinates of Points
    print("Unify Coordinate")
    #Get 3D coordinates of Points
    coordinate_points=[]
    for point in covisibility:
        for i in range(len(point)):
            if point[i]!=-1.0:
                u,v=sift_result[i][0][int(point[i])].pt
                u=np.float64(u)
                v=np.float64(v)

                # Normalized image plane -> (u, v, 1) * z = zu, zv, z
                z = np.asarray(depth[i], dtype=np.float64)[np.int32(v)][np.int32(u)] / depth_scaling_factor  # in mm distance
                x = (u - c_x) * z / f_x
                y = (v - c_y) * z / f_y
                relative_3d = np.array([x,y,z])

                global_3d=inverse_global_pose[i][:3,:3]@relative_3d+inverse_global_pose[i][:3,3]
                coordinate_points.append(global_3d)

                break
    coordinate_points=np.array(coordinate_points)
    print("Done")
    print("Make File")

    #Make TXT File for Bundle Adjustment
    f=open('problem.txt', 'w')

    # number of Cameras, number of 3D points, number of observation
    data="{}\t{}\t{}\n".format(covisibility.shape[1],len(covisibility),observations)
    f.write(data)

    # Camera Index, Point index, observation image coordinate
    index=0
    for co in covisibility:
        for i in range(covisibility.shape[1]):
            if co[i]==-1:
                continue
            point_2d=sift_result[i][0][int(co[i])].pt
            data="{}\t{}\t{}\t{}\n".format(i,index,point_2d[0],point_2d[1])
            f.write(data)
        index+=1


    # Camera Params -> Rotation Vectors, Translation, Focal length, cx, cy

    for pose in forward_global_pose:
        r=R.from_matrix(pose[:3,:3])
        for vec in r.as_rotvec():
            f.write("{}\n".format(vec))

        t=pose[:3,3]
        for vec in t:
            f.write("{}\n".format(vec))



    # 3D Point
    for point in coordinate_points:
        for p in point:
            f.write("{}\n".format(p))

    f.close()
    print("Done")




####################################################################################################################################################################
    print("Start Bundle Adjustment")
    for i in range(1):
        ####################################################################################################################################################################
        # Reprojection Checking

        camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data("problem.txt")

        observations=points_2d.shape[0]

        # Print information
        n_cameras = camera_params.shape[0]
        n_points = points_3d.shape[0]

        n = 6 * n_cameras + 3 * n_points
        m = 2 * points_2d.shape[0]

        points_proj, points_2d = fun2(camera_indices, point_indices, camera_params, points_3d, points_2d)
        reprojection_error = (points_proj - points_2d) ** 2
        reprojection_error = np.sum(reprojection_error, axis=1)
        reprojection_error = np.sqrt(reprojection_error)
        reprojection_error = np.sum(reprojection_error, axis=0)
        befores = reprojection_error / observations



        for k in range(1):
            # RT,3D Point Optimization
            print("RT optimizing started")
            A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices,observations)
            x0 = np.hstack((camera_params.ravel(),points_3d.ravel(),points_2d.ravel()))
            res1 = least_squares(fun, x0,jac_sparsity=A, verbose=0, x_scale='jac', ftol=1e-4, method='trf',
                                args=(n_cameras,n_points,observations,camera_indices, point_indices))

            camera_params = res1.x[:6*n_cameras].reshape((n_cameras, 6))
            points_3d = res1.x[6*n_cameras:6*n_cameras+3*n_points].reshape((n_points, 3))

            # Keypoint 2D Optimization
            print("Keypoint Optimizing started")
            A=keypoint_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices, observations)
            x0 = np.hstack((camera_params.ravel(),points_3d.ravel(),points_2d.ravel()))
            res2 = least_squares(fun, x0, jac_sparsity=A,verbose=0, x_scale='jac', ftol=1e-4, method='trf',
                                args=(n_cameras,n_points,observations,camera_indices, point_indices))

            points_2d=res2.x[6*n_cameras+n_points*3:].reshape((observations,2))


            points_proj, points_2d = fun2(camera_indices, point_indices, camera_params,points_3d,points_2d)
            reprojection_error=(points_proj-points_2d)**2
            reprojection_error=np.sum(reprojection_error,axis=1)
            reprojection_error=np.sqrt(reprojection_error)
            reprojection_error=np.sum(reprojection_error,axis=0)
            afters = reprojection_error / observations


        for i in range(n_cameras):
            index = []
            for k in range(len(camera_indices)):
                if camera_indices[k] == i:
                    index.append(k)

            pp = points_proj[index]
            p2 = points_2d[index]

            orgin = []
            reproject = []

            for p in pp:
                reproject.append((int(p[0]), int(p[1])))
            for p in p2:
                orgin.append((int(p[0]), int(p[1])))

            img = image[i]

            # 색상을 BGR 형식으로 지정합니다.
            color = (255, 0, 0)  # Red

            # (x, y) 좌표를 리스트 형태로 지정합니다.
            coordinates = orgin

            # 각 좌표에 원을 그립니다.
            radius = 3
            for coord in coordinates:
                cv2.circle(img, coord, radius, color, -1)

            # 색상을 BGR 형식으로 지정합니다.
            color = (0, 0, 255)  # Red

            # (x, y) 좌표를 리스트 형태로 지정합니다.
            coordinates = reproject

            # 각 좌표에 원을 그립니다.
            radius = 2
            for coord in coordinates:
                cv2.circle(img, coord, radius, color, -1)

            cv2.imwrite('./2_BA/results/result_bundle{}.png'.format(i), img)


        camera = res1.x[:n_cameras * 6]
        result = None
        for i in range(n_cameras):
            c1 = camera[i * 6:6 * (i + 1)]

            R_1 = R.from_rotvec(c1[:3])
            t_1 = c1[3:6]
            trans1 = np.identity(4)
            trans1[:3, :3] = R_1.as_matrix()
            trans1[:3, 3] = t_1

            trans1 = np.linalg.inv(trans1)

            if result == None:
                copy1 = copy.deepcopy(pcd[i])
                result = copy1.transform(trans1)
            else:
                copy1 = copy.deepcopy(pcd[i])
                result += copy1.transform(trans1)
        o3d.visualization.draw_geometries([result])     
    print("Before: {} After: {}".format(befores, afters))


