#!/usr/bin/env python2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from ldriver.steering.pedestrian_detection import redline_detected
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()

def callback(image_data):
    try:
        image = bridge.imgmsg_to_cv2(image_data, "bgr8")
    except CvBridgeError as e:
        print(e)
    detection = Bool()
    
    if redline_detected(image, 0.26):
        detection.data = True
    rospy.logdebug('Redline: {}'.format(detection.data))
    detection_pub.publish(detection)

if __name__ == "__main__":
    rospy.init_node("RoadDetectionRedline", log_level=rospy.DEBUG)
    detection_pub = rospy.Publisher("/redline", Bool, queue_size=10)
    rospy.sleep(0.5)
    image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, callback, queue_size=1)
    
    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        rate.sleep()