import numpy as np
import cv2 #freetype

mode = 0

#创建回调函数
def OnMouseAction(event,x,y,flags,param):
    global x1, y1
    img = img1

    if mode == 0 and event == cv2.EVENT_LBUTTONDOWN:
        print("左键点击")
        cv2.line(img,(0,0),(x,y),(255,255,0),2)

    if mode == 1 and event == cv2.EVENT_LBUTTONDOWN:
        print("左键点击1")
        x1, y1 = x, y
        print(x1, y1)
    elif mode == 1 and event==cv2.EVENT_MOUSEMOVE and flags ==cv2.EVENT_FLAG_LBUTTON:
        print("左鍵拖曳1")
        cv2.rectangle(img2,(x1,y1),(x,y),(0,255,0),-1)



img1 = np.zeros((500,500,3),np.uint8)
img2= np.zeros((500,500,3),np.uint8)
cv2.putText(img1,'1',(200,200),cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0,0))
cv2.putText(img2,'2',(200,200),cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0,0))
cv2.namedWindow('image1')
cv2.namedWindow('image2')
cv2.setMouseCallback('image1',OnMouseAction)

while(1):
    cv2.imshow('image1',img1)
    cv2.imshow('image2', img2)
    # cbf()
    k=cv2.waitKey(1)
    if k==ord('l'):
        mode = 0
    elif k==ord('t'):
        mode = 1
    elif k==ord('q'):
        break
cv2.destroyAllWindows()
