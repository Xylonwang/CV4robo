import numpy as np
from numpy.linalg import inv

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