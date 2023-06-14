import numpy as np
import open3d as o3d
import cv2
from registration import *
from collections import Counter
import random
import copy




"""
# Pyrealsense
depth_scaling_factor = 999.99
f_x = 618.212
f_y=617.673
c_x = 323.716
c_y = 256.977
k1=0
k2=0
"""

"""
#Calibration
depth_scaling_factor = 999.99
f_x = 615.85642
f_y=614.153577
c_x = 312.144210
c_y = 269.969336
k1=0.169048
k2=-0.263841
"""

"""
#Internet
depth_scaling_factor = 999.99
f_x = 597.522  ## mm
f_y = 597.522
c_x = 312.885
c_y = 239.870
k1=0
k2=0
"""

"""
#ChatGPT
depth_scaling_factor = 999.99
f_x = 616.024  
f_y=616.151
c_x = 321.247
c_y = 238.376
k1=-0.1784
k2=0.1365
"""


# Pyrealsense
depth_scaling_factor = 999.99
f_x = 618.212
f_y=617.673
c_x = 323.716
c_y = 256.977
k1=0
k2=0


def get_boundary(source_pcd):



    x_min = np.min(np.asarray(source_pcd.points)[:, 0])
    x_max = np.max(np.asarray(source_pcd.points)[:, 0])
    y_min = np.min(np.asarray(source_pcd.points)[:, 1])
    y_max = np.max(np.asarray(source_pcd.points)[:, 1])

    x_min_idx = np.where(np.asarray(source_pcd.points)[:, 0] == x_min)
    x_max_idx = np.where(np.asarray(source_pcd.points)[:, 0] == x_max)
    y_min_idx = np.where(np.asarray(source_pcd.points)[:, 1] == y_min)
    y_max_idx = np.where(np.asarray(source_pcd.points)[:, 1] == y_max)

    u_min = x_min * f_x / (np.asarray(source_pcd.points)[x_min_idx][0][2]) + c_x
    u_max = x_max * f_x / (np.asarray(source_pcd.points)[x_max_idx][0][2]) + c_x
    v_min = y_min * f_y / (np.asarray(source_pcd.points)[y_min_idx][0][2]) + c_y
    v_max = y_max * f_y / (np.asarray(source_pcd.points)[y_max_idx][0][2]) + c_y

    return u_min, u_max, v_min, v_max


def preprocess_point_cloud(pcd, voxel_size):

    '''
    :param pcd: Point cloud dataset
    :param voxel_size: Voxel size of dataset
    :return: downsampled point cloud
    '''
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh




# Output: 3D coordinate, forward relative pose, inverse relative pose,
def Pose_Estimation(sift_result1,sift_result2,img1, img2, depth_img1, depth_img2, source_pcd, target_pcd,distance_ratio,samples=None,tol=1e-6):

    # Read image from path
    imgL=cv2.imread(img1)
    imgR=cv2.imread(img2)
    depthL=np.array(o3d.io.read_image(depth_img1),np.float32)
    depthR=np.array(o3d.io.read_image(depth_img2),np.float32)

    # Clip depth value
    threshold=1500 #1.5m limit
    left_idx=np.where(depthL>threshold)
    right_idx=np.where(depthR>threshold)
    depthL[left_idx]=threshold
    depthR[right_idx]=threshold


    result=SIFT_Transformation(sift_result1,sift_result2,imgL,imgR,depthL,depthR,source_pcd,target_pcd,distance_ratio)

    if result==None:
        return None

    pts1_3d, pts2_3d, pts1, pts2, good_matches=result

    max_iteration=10
    indices=np.arange(pts1_3d.shape[0]).tolist()
    bestErr=1e10

    temp_R_T=None
    best_inlier=None
    bestFit=None
    best_matches=[]

    n=samples
    for _ in range(max_iteration):
        samples = random.sample(indices, n)

        R_t=np.array(estimate_transformation(pts1_3d[samples],pts2_3d[samples]))
        R_temp=R_t[:3,:3]
        t_temp=R_t[:3,3].T
        transformed=(np.dot(R_temp, pts1_3d.T).T)+t_temp
        error=(transformed-pts2_3d)**2
        error=np.sum(error,axis=1)
        error=np.sqrt(error)
        """int(pts1_3d.shape[0]*0.7)"""
        inlier_indices=np.where(error<tol)[0]
        if inlier_indices.shape[0]>=int(pts1_3d.shape[0]*0.7):
            better_R_t = np.array(estimate_transformation(pts1_3d[inlier_indices], pts2_3d[inlier_indices]))
            R_temp=better_R_t[:3,:3]
            t_temp=better_R_t[:3,3].T
            transformed=(np.dot(R_temp, pts1_3d[inlier_indices].T).T)+t_temp
            error=(transformed-pts2_3d[inlier_indices])**2
            error=np.sum(error,axis=1)
            error=np.sum(np.sqrt(error))/error.shape[0]

            if error<bestErr:
                bestFit=better_R_t
                bestErr=error
                best_inlier=inlier_indices

    if bestErr==1e10:
        return None

    for i in best_inlier:
        best_matches.append(good_matches[i])

    return bestFit,best_matches

# Output: 3D coordinate, forward relative pose, inverse relative pose,
def SIFT_Transformation(sift_result1,sift_result2,imgL, imgR, depthL, depthR, source_pcd, target_pcd, distance_ratio):


    # Clip depth value
    threshold=1500 #1.5m limit
    left_idx=np.where(depthL>threshold)
    right_idx=np.where(depthR>threshold)
    depthL[left_idx]=threshold
    depthR[right_idx]=threshold

    # Find keypoints and descriptors using SIFT
    kp1,des1=sift_result1
    kp2,des2=sift_result2

    # BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)
    matches_forward = bf.knnMatch(des1, des2, k=2)
    matches_inverse = bf.knnMatch(des2, des1, k=2)

    # Need to draw only good matches, so create a mask
    good_matches_forward=[]
    source_x_min, source_x_max, source_y_min, source_y_max = get_boundary(source_pcd)
    target_x_min, target_x_max, target_y_min, target_y_max = get_boundary(target_pcd)

    for i, (m, n) in enumerate(matches_forward):
        if m.distance < distance_ratio*n.distance:
            if (kp1[m.queryIdx].pt[0] >= source_x_min and kp1[m.queryIdx].pt[0] <= source_x_max):
                if (kp1[m.queryIdx].pt[1] >= source_y_min and kp1[m.queryIdx].pt[1] <= source_y_max):
                    if (kp2[m.trainIdx].pt[0] >= target_x_min and kp2[m.trainIdx].pt[0] <= target_x_max):
                        if (kp2[m.trainIdx].pt[1] >= target_y_min and kp2[m.trainIdx].pt[1] <= target_y_max):
                            good_matches_forward.append(m)

    good_matches_inverse=[]
    source_x_min, source_x_max, source_y_min, source_y_max = get_boundary(target_pcd)
    target_x_min, target_x_max, target_y_min, target_y_max = get_boundary(source_pcd)

    for i, (m, n) in enumerate(matches_inverse):
        if m.distance < distance_ratio*n.distance:
            if (kp2[m.queryIdx].pt[0] >= source_x_min and kp2[m.queryIdx].pt[0] <= source_x_max):
                if (kp2[m.queryIdx].pt[1] >= source_y_min and kp2[m.queryIdx].pt[1] <= source_y_max):
                    if (kp1[m.trainIdx].pt[0] >= target_x_min and kp1[m.trainIdx].pt[0] <= target_x_max):
                        if (kp1[m.trainIdx].pt[1] >= target_y_min and kp1[m.trainIdx].pt[1] <= target_y_max):
                            good_matches_inverse.append(m)

    good_matches=[]
    good_matches_v=[]
    pts1=[]
    pts2=[]
    kp1_1=[]
    kp2_1=[]
    for m_f in good_matches_forward:
        for m_i in good_matches_inverse:
            if m_f.queryIdx==m_i.trainIdx and m_f.trainIdx==m_i.queryIdx:
                good_matches_v.append([m_f])
                good_matches.append(m_f)
                pts1.append(kp1[m_f.queryIdx].pt)
                pts2.append(kp2[m_f.trainIdx].pt)
                break

    """
    img_matched = cv2.drawMatchesKnn(imgL, kp1, imgR, kp2, good_matches_v, None, matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0), flags=2)
    cv2.imshow('img_matched', img_matched)

    cv2.waitKey(0)
    """


    if len(good_matches)<3:
        return None

    # Set array for keypoints
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    pts1_3d = []
    pts2_3d = []


    for i in range(pts1.shape[0]):
        u = np.float64(pts1[i][0])
        v = np.float64(pts1[i][1])

        z = np.asarray(depthL, dtype=np.float64)[np.int32(v)][np.int32(u)] / depth_scaling_factor  # in mm distance
        x = (u - c_x) * z / (f_x)
        y = (v - c_y) * z / (f_y)
        pts1_3d = np.append(pts1_3d, np.array([x, y, z], dtype=np.float32))

    for i in range(pts2.shape[0]):
        u = np.float64(pts2[i][0])
        v = np.float64(pts2[i][1])

        z = np.asarray(depthR, dtype=np.float64)[np.int32(v)][np.int32(u)] / depth_scaling_factor  # in mm distance
        x = (u - c_x) * z / (f_x)
        y = (v - c_y) * z / (f_y)
        pts2_3d = np.append(pts2_3d, np.array([x, y, z], dtype=np.float32))



    pts1_3d = pts1_3d.reshape(-1, 3)
    pts2_3d = pts2_3d.reshape(-1, 3)

    return pts1_3d,pts2_3d,pts1,pts2,good_matches

