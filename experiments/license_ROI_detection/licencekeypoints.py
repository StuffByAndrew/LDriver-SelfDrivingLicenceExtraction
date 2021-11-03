#!/usr/bin/env python
from logging import error
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from cv2.xfeatures2d import SIFT_create
import sys
import numpy as np


def proccess_image(data):
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='mono8')
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray,None)

    img=cv2.drawKeypoints(gray,kp)
    cv2.imshow('a', img)
    cv2.waitKey(1)
    

if __name__ == '__main__':
    rospy.init_node('line_follower', anonymous=True)

    sift = SIFT_create()
    bridge = CvBridge()

    rospy.Subscriber("/R1/pi_camera/image_raw", Image, proccess_image)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
