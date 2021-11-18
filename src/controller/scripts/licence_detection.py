#!/usr/bin/env python2
from os import EX_CANTCREAT
from ldriver.licence.detection import find_licence
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy

def proccess_image(data):
    orig_img = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
    try:
        plate = find_licence(orig_img)
        cv2.imshow('plate', plate)
        cv2.waitKey(1)
    except:
        pass

if __name__ == '__main__':
    rospy.init_node('licensedriver', anonymous=True)
    bridge = CvBridge()
    rospy.Subscriber("/R1/pi_camera/image_raw", Image, proccess_image)
    rospy.spin()