import cv2
import time
import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
from lane_detection import get_roadcolor_center, get_right_lines_center, mask_rectangle, get_bottom_right_line_center, horizontal_distance_from_line
from pedestrian_detection import pedestrian_crossing, redline_detected, hsv_threshold, dilate_erode
bridge = CvBridge()
"""
Car: 
lower: [0, 0, 105]
upper: [0, 0, 200]
"""
def car_motion_detection(current_image_input, previous_image):
    current_image = hsv_threshold(current_image_input, lh=0, ls=0, lv=105, uh=0, us=0, uv=200)
    current_image = cv2.GaussianBlur(current_image, (21,21), 0)
    current_image = dilate_erode(current_image, 0, 5)
    if previous_image is None:
        previous_image = current_image
    difference = cv2.absdiff(current_image, previous_image)
    _, thresh = cv2.threshold(difference,55,255,cv2.THRESH_BINARY)
    previous_image = current_image
    return np.count_nonzero(thresh), current_image
    
history = [0,0,0]
previous_image = None
car_was_detected = False
def robot_should_move(current_image, threshold):
    global previous_image
    nonzero, previous_image = car_motion_detection(current_image, previous_image) 
    return nonzero < threshold
    
def callback(image_data):
    global previous_image
    try:
        image = bridge.imgmsg_to_cv2(image_data, "bgr8")
    except CvBridgeError as e:
        print(e)
    #current_image = cv2.cvtColor(current_image_input, cv2.COLOR_BGR2GRAY)
    current_image = hsv_threshold(image, lh=0, ls=0, lv=105, uh=0, us=0, uv=200)
    current_image = cv2.GaussianBlur(current_image, (21,21), 0)
    #current_image = dilate_erode(current_image, 0, 5)

    if previous_image is None:
        previous_image = current_image
    difference = cv2.absdiff(current_image, previous_image)
    _, thresh = cv2.threshold(difference,55,255,cv2.THRESH_BINARY)
    previous_image = current_image
    # cv2.imshow("thresh", image)
    # cv2.waitKey(1)
    cv2.imshow("thresh", image)
    cv2.waitKey(1)

def rotate_left(interval):
    command = Twist()
    command.angular.z = 0.62    
    move_pub.publish(command)
    rospy.sleep(interval)

if __name__ == "__main__":
    rospy.init_node("lane_following", anonymous=True)
    move_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
    rospy.sleep(0.5)
    #rotate_left(1.5)
    image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, callback, queue_size=1)
    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        rate.sleep()