import numpy as np
import open3d as o3d
import cv2
import copy
from registration import *
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import numpy as np
from scipy.spatial.transform import Rotation as R
from SIFT import *


# Pyrealsense
depth_scaling_factor = 999.99
f_x = 618.212
f_y=617.673
c_x = 323.716
c_y = 256.977
k1=0
k2=0

def get_covisibility(sift_result,img,depth,pcd,distance_ratio,samples=3,tol=1e-2):
    holy=0
    covisibility_graph={}
    matches=None
    issue=0
    for i in range(1,len(sift_result)):
        for j in range(i-1,-1,-1):

            result = Pose_Estimation(
                sift_result[i], sift_result[j], img[i], img[j], depth[i], depth[j],
                pcd[i],
                pcd[j],samples=samples, distance_ratio=distance_ratio,tol=tol)
            if result==None:
                continue
            matches=result[1]


            for m in matches:
                bool = False
                # 같은 점을 바라보는 다른 카메라 시점의 keypoing 그룹이 있는지 확인
                for key, value in covisibility_graph.items():
                    if (j, m.trainIdx) in value and (i, m.queryIdx) not in value:

                        if any([i == v[0] for v in value]):
                            issue+=1
                            covisibility_graph.pop(key)
                            break
                        value.append((i, m.queryIdx))
                        bool = True
                        break
                # 만약 2 시점에서 바라보는 점이 최초라면 그래프 확장
                if bool == False:
                    covisibility_graph[len(covisibility_graph)] = [(i, m.queryIdx), (j, m.trainIdx)]

    covisibility = np.ones((len(covisibility_graph), len(sift_result))) * -1
    j=0
    for key, value in covisibility_graph.items():
        for v in value:
            covisibility[j][v[0]] = v[1]
        j+=1
    o = 0
    for co in covisibility:
        for c in co:
            if c != -1:
                o += 1
    far=0
    for co in covisibility:
        for i in range(len(sift_result)-1):
            if co[i]!=-1 and co[i+1]!=-1:
                far+=1
                break
    print(issue,far,covisibility.shape[0],"{:.3f}%".format(100*far/covisibility.shape[0]))
    return covisibility, o



def get_global_poses(relative_poses):

    global_poses=[]
    base_corrdinate=np.identity(4)
    global_poses.append(base_corrdinate)

    for i in range(len(relative_poses)):
        pre_pose=global_poses[i]
        relative_pose=relative_poses[i]

        new_pose=relative_pose@pre_pose
        global_poses.append(new_pose)

    return global_poses



def read_bal_data(file_name):
    with open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 6)
        for i in range(n_cameras * 6):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d

from scipy.spatial.transform import Rotation as R

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    """
    for i in range(points.shape[0]):
        rotation = R.from_rotvec(rot_vecs[i]).as_matrix()
        rotated=rotation@points[i].T
        points[i]=rotated.T

    return points

def rotate2(points, rot_vecs):


    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) *v



def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""


    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]

    for i in range(points_proj.shape[0]):
            points_proj[i][0]/=points_proj[i][2]
            points_proj[i][1]/=points_proj[i][2]

    points_proj[:,0]=f_x*points_proj[:,0]+c_x
    points_proj[:,1]=f_y*points_proj[:,1]+c_y
    points_proj=points_proj[:,:2]

    return points_proj


def fun(params, n_cameras, n_points, observations, camera_indices, point_indices):
    camera_params = params[:6*n_cameras].reshape((n_cameras, 6))
    points_3d = params[6*n_cameras:6*n_cameras+3*n_points].reshape((n_points, 3))
    points_2d = params[6*n_cameras+n_points*3:].reshape((observations, 2))

    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    residual = (points_proj - points_2d).ravel()

    return residual


def fun2(camera_indices, point_indices, camera_params,points_3d,points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """

    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    residual = (points_proj - points_2d).ravel()


    return points_proj,points_2d



def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices,observations):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3 + observations*2
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1
    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
    observation_indices=np.arange(observations)
    for s in range(2):
        A[2 * i, n_cameras * 6 + n_points * 3+observation_indices*2 + s] = 0
        A[2 * i + 1, n_cameras * 6 + n_points * 3+observation_indices*2 + s] = 0

    return A
def keypoint_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices, observations):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3+ observations*2
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 0
        A[2 * i + 1, camera_indices * 6 + s] = 0
    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 0
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 0
    observation_indices = np.arange(observations)
    for s in range(2):
        A[2 * i, n_cameras * 6 + n_points * 3 + observation_indices * 2 + s] = 1
        A[2 * i + 1, n_cameras * 6 + n_points * 3 + observation_indices * 2 + s] = 1
    return A