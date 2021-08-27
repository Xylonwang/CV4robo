import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space


def dlt_homography(I1pts, I2pts):
    """
    Using Direct Linear Transform find the perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    -----------
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    # row and col of image1
    I1pts_row = I1pts.shape[0]
    I1pts_col = I1pts.shape[1]
    # define the DLT matrix A
    A_row = I1pts_row * I1pts_col
    A_col = 9
    # initialize A
    A = np.zeros(shape=(A_row, A_col))
    # load the coordinates to A
    for k in range(I1pts.shape[1]):
        x = I1pts[0, k]
        y = I1pts[1, k]
        u = I2pts[0, k]
        v = I2pts[1, k]
        A[2*k, :] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
        A[2*k+1, :] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
    # calculate the zero space of matrix A, this will return
    # a general solution of H
    H = null_space(A)
    # normalize the H
    H = H/H[8, 0]
    # reshape it to matrix form
    H = H.reshape(3, 3)
    return H, A
