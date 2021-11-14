import cv2 
import numpy as np

uh = 0
us = 0
uv1 = 125
lh = 0
ls = 0
lv1 = 90
lower_hsv1 = np.array([lh,ls,lv1])
upper_hsv1 = np.array([uh,us,uv1])

lv2 = 155
uv2 = 210
lower_hsv2 = np.array([lh,ls,lv2])
upper_hsv2 = np.array([uh,us,uv2])

def hsv_threshold(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.medianBlur(hsv,5)
    msk1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
    msk2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
    return cv2.bitwise_or(msk1, msk2)

def display_hsv_threshold(image_file):
    cv2.imshow("HSV Threshold", hsv_threshold(image_file))
    cv2.waitKey(0)