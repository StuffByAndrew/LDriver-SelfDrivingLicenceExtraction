#!/usr/bin/env python2
import rospy
from sensor_msgs.msg import Image

def callback(image_data):
    detection_pub.publish(image_data)

if __name__ == "__main__":
    rospy.init_node("Image2", log_level=rospy.DEBUG)
    detection_pub = rospy.Publisher("/R1/pi_camera/image_raw2", Image, queue_size=1)
    rospy.sleep(0.5)
    image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, callback, queue_size=1)
    
    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        rate.sleep()