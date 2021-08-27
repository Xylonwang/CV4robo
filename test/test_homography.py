import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from imageio import imread, imwrite
import cv2
from geometric_image_transformations.dlt_homography import dlt_homography
from color_conversions.bilinear_interpolation import bilinear_interp
from color_conversions.alpha_blending import alpha_blend
from interface.click_coord import on_EVENT_LBUTTONDOWN


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


if __name__ == "__main__":
    ibg = imread('../data/images/DSC_1188.jpg')

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

    ifg = imread('../data/images/DSC_1190.jpg')

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
