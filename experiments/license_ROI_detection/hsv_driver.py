#!/usr/bin/env python

import numpy as np
import cv2 as cv
import time
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class hsvDisplay:

    # Convert BGR to HSV
    # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    uh = 130
    us = 255
    uv = 255
    lh = 110
    ls = 50
    lv = 50
    lower_hsv = np.array([lh,ls,lv])
    upper_hsv = np.array([uh,us,uv])

    # Threshold the HSV image to get only blue colors
    window_name = "HSV Calibrator"
    cv.namedWindow(window_name)

    def nothing(x):
        print("Trackbar value: " + str(x))
        pass

    # create trackbars for Upper HSV
    cv.createTrackbar('UpperH',window_name,0,255,nothing)
    cv.setTrackbarPos('UpperH',window_name, uh)

    cv.createTrackbar('UpperS',window_name,0,255,nothing)
    cv.setTrackbarPos('UpperS',window_name, us)

    cv.createTrackbar('UpperV',window_name,0,255,nothing)
    cv.setTrackbarPos('UpperV',window_name, uv)

    # create trackbars for Lower HSV
    cv.createTrackbar('LowerH',window_name,0,255,nothing)
    cv.setTrackbarPos('LowerH',window_name, lh)

    cv.createTrackbar('LowerS',window_name,0,255,nothing)
    cv.setTrackbarPos('LowerS',window_name, ls)

    cv.createTrackbar('LowerV',window_name,0,255,nothing)
    cv.setTrackbarPos('LowerV',window_name, lv)

    #cv.createButton('Save HSV', window_name)

    font = cv.FONT_HERSHEY_SIMPLEX
    def __init__(self):
        self.bridge = CvBridge()

    def callback(self, data):
        
        img = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        img = cv.medianBlur(img,5)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # Threshold the HSV image to get only blue colors
        mask = cv.inRange(hsv, self.lower_hsv, self.upper_hsv)
        cv.putText(mask,'Lower HSV: [' + str(self.lh) +',' + str(self.ls) + ',' + str(self.lv) + ']', (10,30), self.font, 0.5, (200,255,155), 1, cv.LINE_AA)
        cv.putText(mask,'Upper HSV: [' + str(self.uh) +',' + str(self.us) + ',' + str(self.uv) + ']', (10,60), self.font, 0.5, (200,255,155), 1, cv.LINE_AA)
        
        cv.imshow('a', img)
        # cv.waitKey(1)
        
        # get current positions of Upper HSV trackbars
        self.uh = cv.getTrackbarPos('UpperH',self.window_name)
        self.us = cv.getTrackbarPos('UpperS',self.window_name)
        self.uv = cv.getTrackbarPos('UpperV',self.window_name)
        self.upper_blue = np.array([self.uh,self.us,self.uv])
        # get current positions of Lower HSCV trackbars
        self.lh = cv.getTrackbarPos('LowerH',self.window_name)
        self.ls = cv.getTrackbarPos('LowerS',self.window_name)
        self.lv = cv.getTrackbarPos('LowerV',self.window_name)
        self.upper_hsv = np.array([self.uh,self.us,self.uv])
        self.lower_hsv = np.array([self.lh,self.ls,self.lv])

    

if __name__ == '__main__':
    rospy.init_node('licensedriver', anonymous=True)
    hsv = hsvDisplay()
    rospy.Subscriber("/R1/pi_camera/image_raw", Image, hsv.callback)
    rospy.spin()
    cv.destroyAllWindows()