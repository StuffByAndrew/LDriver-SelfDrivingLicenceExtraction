#!/usr/bin/env python2
from os import EX_CANTCREAT
from ldriver.licence.detection import LicencePlate
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy
from ldriver.licence.ocr import LicenceOCR
import tensorflow as tf
import time

def process_image(data):
    cv_img = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
    print(cv_img.dtype)
    lp = LicencePlate(cv_img) 
    if lp.valid:
        cv2.imshow('plate', lp.img)
        cv2.waitKey(1)
        # graph = tf.get_default_graph()
        # with graph.as_default():
        res = locr.read_licence(lp)
        print(res)

if __name__ == '__main__':
    rospy.init_node('licensedriver', anonymous=True)
    locr = LicenceOCR()
    orig_img = cv2.imread('experiments/test_images/74.png')
    # lp = LicencePlate(orig_img)
    # if lp.valid:
    #     res = locr.read_licence(lp)
    #     print(res)
    #     cv2.imshow('a', lp.img)
    #     cv2.waitKey(0)
    bridge = CvBridge()
    rospy.Subscriber("/R1/pi_camera/image_raw", Image, process_image)
    rospy.spin()