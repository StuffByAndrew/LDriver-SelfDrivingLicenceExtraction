#!/usr/bin/env python2
import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from ldriver.steering.road_detection import detect_gl
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

def green_line(image_data):
    try:
        image = bridge.imgmsg_to_cv2(image_data, "bgr8")
    except CvBridgeError as e:
        print(e)
    detection = Bool()
    line, thresh = detect_gl(image)
    rospy.loginfo(str(line))
    cv2.imshow('testing', thresh)
    cv2.waitKey(1)
    if len(line):
        detection.data = True
    detection_pub.publish(detection)

if __name__ == '__main__':
    rospy.init_node("RoadDetectionGreenline")
    detection_pub = rospy.Publisher("/RoadDetection/greenline", Bool, queue_size=10)
    rospy.sleep(0.5)
    image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, green_line, queue_size=1)
    
    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        rate.sleep()