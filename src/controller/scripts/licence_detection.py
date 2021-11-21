#!/usr/bin/env python2
from os import EX_CANTCREAT
from ldriver.licence.detection import LicencePlate
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy
from ldriver.licence.ocr import LicenceOCR

def proccess_image(data):
    orig_img = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
    lp = LicencePlate(orig_img)
    print(locr.read_licence(lp))
    if lp.valid:
        cv2.imshow('plate', lp.img)
        cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('licensedriver', anonymous=True)
    locr = LicenceOCR()
    bridge = CvBridge()
    rospy.Subscriber("/R1/pi_camera/image_raw", Image, proccess_image)
    rospy.spin()