import cv2
import numpy as np

# image path
# img = cv2.imread('../data/images/rover_forward_navcam.png')


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


if __name__ == "__main__":
    # image path
    img = cv2.imread('../data/images/rover_forward_navcam.png')
    xpts = []
    ypts = []
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    print(xpts, ypts)
