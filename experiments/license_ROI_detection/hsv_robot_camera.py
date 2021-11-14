import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from hsv_threshold import hsv_threshold

def process_image(data):
    image = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
    cv2.imshow("HSV Threshold", hsv_threshold(image))
    cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('line_follower', anonymous=True)
    bridge = CvBridge()
    rospy.Subscriber("/R1/pi_camera/image_raw", Image, process_image)
    rospy.spin()