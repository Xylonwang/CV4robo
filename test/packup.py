import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from imageio import imread, imwrite
import cv2
from scipy.linalg import null_space
import os
'''

'''
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

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """

    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')

    # --- FILL ME IN ---

    # Code goes here...

    # Define the coordinates
    # because the relation between coordinates and row number or
    # col number of pixels in image
    y = pt[0, 0]
    x = pt[1, 0]
    # define the four pixel point of the bilinear-interpo operation
    x1 = int(np.floor(x))
    x2 = int(np.ceil(x))
    y1 = int(np.floor(y))
    y2 = int(np.ceil(y))
    # to avoid pick pixel out of bottom and right bounds of image
    if x >= I.shape[0]-1:
        x1 = x1 - 1
        x2 = x2 - 1
        x = x-1

    if y >= I.shape[1]-1:
        y1 = y1 - 1
        y2 = y2 - 1
        y = y-1
    # get the intensity of 4 points
    Q12 = I[x1, y1]
    Q22 = I[x1, y2]
    Q21 = I[x2, y2]
    Q11 = I[x2, y1]
    Q = np.array([[Q11, Q12], [Q21, Q22]])
    # to simplify the expression of the calculation
    # define to variables
    v = x-x1  # x2 - x, 1-u: x - x1
    u = y2-y  # y2 - y, 1-v: y - y1

    X = np.array([u, 1 - u])
    Y = np.array([v, 1 - v]).T

    # b = Q12*(1-u)*(1-v)+Q22*(1-u)*v+Q11*u*(1-v)+Q21*u*v

    # perform the calculation
    b = np.dot(np.dot(X, Q), Y)
    # because intensity need to be integer, round b
    b = np.around(b)
    # convert to uint8
    b = b.astype(np.uint8)

    # ------------------

    return b


def alpha_blend(Ifg, Ibg, alpha):
    """
    Performs alpha blending of two images.

    The alpha parameter controls the amount of blending between the foreground
    image and the background image. The background image is assumed to be fully
    opaque.

    Note that images are (usually) integer-valued functions, therefore the
    image you return must be integer-valued (use round()).

    This function should accept both greyscale and RGB images, and return an
    image of the same type.

    Parameters:
    -----------
    Ifg    - Greyscale or RGB foreground image, 8-bit np.array (i.e., uint8).
    Ibg    - Greyscale or RGB background image, 8-bit np.array (i.e., uint8).
    alpha  - Floating point blending parameter, [0, 1].

    Returns:
    --------
    Ia  - Alpha-bended image (*must be* same size as original images).
    """

    if Ifg.shape != Ibg.shape:
        raise ValueError('Input images are different sizes.')

    #--- FILL ME IN ---
    # Code goes here...

    # just calculate the new intensity directly
    Ia = np.around(alpha * Ifg + (1 - alpha) * Ibg)

    # convert to uint8
    Ia = Ia.astype(np.uint8)

    #------------------

    return Ia


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global xpts, ypts
    xpts = param[0]
    ypts = param[1]
    img = param[2]
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)

        xpts.append(x)
        ypts.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)


def test_homography(ifg, ibg, ifg_pts, ibg_pts):
    if ifg is None:
        ibg = imread('../data/images/DSC_1188.jpg')
    rectangle_pts = []
    # ibg_pts = np.array([[410, 928, 1240, 64], [192, 192, 792, 786]])
    # ifg_pts = np.array([[2, 898, 898, 2], [2, 2, 601, 601]])
    [H, A] = dlt_homography(ibg_pts, ifg_pts)

    Iover = ibg.copy()
    Iover = np.asarray(Iover)
    Ifg = np.asarray(ifg)
    alpha = 0.7
    polygon = [(0, 0), (0, Ifg.shape[0]), (Ifg.shape[1], Ifg.shape[0]), (Ifg.shape[1], 0)]
    path = Path(polygon)
    # vectorized all the pixel points from Ibg to calculate the new
    # points faster
    xpts = ibg.shape[1]
    ypts = ibg.shape[0]
    xptss = np.arange(xpts)
    yptss = np.arange(ypts)
    xpts_ = xptss.repeat(ypts, axis=0)
    ypts_ = np.tile(yptss, xpts)
    ones = np.ones(xpts_.shape[0])
    # now we have the x, y, 1 to assemble the vector
    points_vector_bg = np.c_[xpts_, ypts_]
    points_vector_bg = np.c_[points_vector_bg, ones]
    # warp calculation
    new_points = np.dot(H, points_vector_bg.T)
    # normalize the new pixel points
    us = new_points[0, :] / new_points[2, :]
    vs = new_points[1, :] / new_points[2, :]
    # assemble the new vector
    new_points = np.c_[us, vs]
    # use contain_points method to see which pixels need to
    # be updated
    t = path.contains_points(new_points)
    # main loop to update the pixels
    for m in range(new_points.shape[0]):
        # if t[m] is true, then this is a pixel that need to be update
        if t[m]:
            # loop through all bands of the image
            for band in range(ibg.shape[2]):
                # coordinate of the new point
                pt = np.array([[new_points[m, 0]], [new_points[m, 1]]])
                # row number and col number of the pixel
                # please note row are y coordinate and col are x coordinate
                # and y axis need to revise
                x = xpts_[m]
                y = ypts_[m]
                # load the new intensity to the pixel
                Iover[y, x, band] = bilinear_interp(Ifg[:, :, band], pt)
        else:
            continue
    Iover = Iover.astype(np.uint8)
    # blending the overlay image with background
    Iover = alpha_blend(Iover, ibg, alpha)

    # print the time consumption
    # print('totally cost', time_end - time_start)
    # plt.imshow(Iover)
    # plt.show()

    return Iover


ibg = imread('./ibg.jpg')

height, width = ibg.shape[:2]
if height > 2000 or width > 2000:
    ibg = cv2.resize(ibg, (int(width/3), int(height/3)), interpolation=cv2.INTER_NEAREST)

xpts = []
ypts = []

img = ibg
param = xpts, ypts, img
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, param)
cv2.imshow("image", ibg)
cv2.waitKey(0)
ibg_pts = np.vstack((xpts, ypts))
# print(ibg_pts)

ifg = imread('./ifg.jpg')

height, width = ifg.shape[:2]
if height > 2000 or width > 2000:
    ifg = cv2.resize(ifg, (int(width / 3), int(height / 3)), interpolation=cv2.INTER_NEAREST)

xpts = []
ypts = []
img = ifg
param = xpts, ypts, img
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, param)
cv2.imshow("image", ifg)
cv2.waitKey(0)
ifg_pts = np.vstack((xpts, ypts))

Ihomo = test_homography(ifg, ibg, ifg_pts, ibg_pts)
plt.imshow(Ihomo)
plt.show()
imwrite('overlay!.jpg', Ihomo)

os.system("pause")
