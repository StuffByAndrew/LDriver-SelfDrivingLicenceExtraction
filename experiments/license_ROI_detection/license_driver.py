import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from hsv_threshold import hsv_threshold as hsv
from plate_contour import rect_contours as rcont
from test import dilate_erode as de
import cv2

def proccess_image(data):
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
    thresh = hsv(cv_image)
    thresh = de(thresh)
    img = rcont(thresh, cv_image)
    cv2.imshow('a', img)
    cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('licensedriver', anonymous=True)
    bridge = CvBridge()
    rospy.Subscriber("/R1/pi_camera/image_raw", Image, proccess_image)
    rospy.spin()