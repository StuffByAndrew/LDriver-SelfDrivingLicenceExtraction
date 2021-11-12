from operator import imod
from hsv_threshold import hsv_threshold as hsv
from plate_contour import rect_contours as rcont
from plate_contour import warp_rect
import cv2
import glob
import numpy as np

def dilate_erode(img):
    kernel5 = np.ones((5,5), np.uint8)
    kernel3 = np.ones((3,3), np.uint8)
 
    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img = cv2.erode(img, kernel3, iterations=2)
    img = cv2.dilate(img, kernel5, iterations=1)
    return img


for img in glob.glob('experiments/test_images/*.png'):
    orig_img = cv2.imread(img)
    img = hsv(orig_img)
    img = dilate_erode(img)
    cnt_img, pts = rcont(img, orig_img)
    if pts.size:
        out=np.float32([[300,0],[0,0],[0,300],[300,300]])
        plate = warp_rect(orig_img, pts, out,size=(300,300))
        cv2.imshow('test', plate)
        cv2.waitKey(0)
    cv2.imshow('test', cnt_img)
    cv2.waitKey(0)