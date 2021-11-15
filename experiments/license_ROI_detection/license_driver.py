import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from hsv_threshold import hsv_threshold as hsv
from plate_contour import rect_contours as rcont
from test import dilate_erode as de
from plate_contour import warp_rect
import cv2
import numpy as np

def proccess_image(data):
    orig_img = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
    img = hsv(orig_img.copy())
    img = de(img)
    cnt_img, pts = rcont(img, orig_img)
    cv2.imshow('contours', cnt_img)
    cv2.waitKey(1)
    if len(pts):
        out = np.float32([[300,0],[0,0],[0,300],[300,300]])
        plate = warp_rect(orig_img, pts, out,size=(300,300))
        cv2.imshow('plate', plate)
        cv2.waitKey(2)

if __name__ == '__main__':
    print('hello')
    rospy.init_node('licensedriver', anonymous=True)
    bridge = CvBridge()
    rospy.Subscriber("/R1/pi_camera/image_raw", Image, proccess_image)
    rospy.spin()