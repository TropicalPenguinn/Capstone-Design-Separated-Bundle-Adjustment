from open3d import *
import numpy as np
import cv2
import open3d as o3d
from scipy.optimize import least_squares


def estimate_transformation(p, p_prime):

    assert len(p) == len(p_prime)
    A=  np.asmatrix(p)
    B=  np.asmatrix(p_prime)
    N = A.shape[0];

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)


    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # AA: [N,3] BB: [N:3]
    # Paper [3,N]

    H = AA.T@BB   # [3,N] [N,3]

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T
    # reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    R = np.array(R)
    t = (np.array(t).T)[0]

    transform = [[R[0][0],R[0][1],R[0][2],t[0]],
                 [R[1][0],R[1][1],R[1][2],t[1]],
                 [R[2][0],R[2][1],R[2][2],t[2]],
                 [0,0,0,1]]

    return transform
