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

<<<<<<< HEAD
def process_image(data):
    cv_img = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
    print(cv_img.dtype)
    lp = LicencePlate(cv_img) 
=======
def proccess_image(data):
    orig_img = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
    lp = LicencePlate(orig_img)
>>>>>>> 283b2a0e8f05a00195f31c9ae9d9c1be179c1510
    if lp.valid:
        locr.read_licence(lp)
        #print(locr.read_licence(lp))
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