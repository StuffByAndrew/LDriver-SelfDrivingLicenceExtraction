import cv2
import time
import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
from lane_detection import get_roadcolor_center, get_right_lines_center, mask_rectangle, get_bottom_right_line_center, horizontal_distance_from_line
from pedestrian_detection import pedestrian_crossing, redline_detected

class PID_Control:
    def __init__(self, KP=0.0, KI=0.0, KD=0.0, I_limit=0.0, rate=0.0):
        self.error, self.prev_error = 0.0, 0.0
        self.KP, self.KI, self.KD = KP, KI, KD
        self.P, self.I, self.D = 0.0, 0.0, 0.0
        self.I_limit = I_limit
        self.rate = rate
    
    def calculate_PID(self, target, measurement):
        self.error = float(target - measurement)
        self.D -= self.KD*(self.error - self.prev_error)/self.rate
        self.P = self.error
        self.I += self.P
        
    def get_error(self):
        return self.KP*self.P + self.KI*self.I + self.KD*self.D

    def __repr__(self):
        return "error: {} P: {}, I: {}, D: {}".format(self.error, self.P, self.I, self.D)

pedestrian_was_crossing = False
previous_image = None

def robot_should_cross(current_image):
    global previous_image
    global pedestrian_was_crossing
    pedestrian_currently_crossing, previous_image = pedestrian_crossing(current_image, previous_image)
    if not pedestrian_currently_crossing and pedestrian_was_crossing:
        pedestrian_was_crossing, previous_image = False, None
        return True
    else:
        pedestrian_was_crossing = pedestrian_currently_crossing
        return False

is_stopped = False
def pid_steering(image_data):
    global is_stopped
    global pedestrian_was_crossing
    global previous_image
    try:
        image = bridge.imgmsg_to_cv2(image_data, "bgr8")
    except CvBridgeError as e:
        print(e)
    
    if redline_detected:
        command = Twist()
        if not is_stopped:
            move_pub.publish(command)
            rospy.sleep(0.5)
            is_stopped = True
        else:
            if robot_should_cross(image):
                command.linear.x = 0.5
                move_pub.publish(command)
                rospy.sleep(1)
                slow_stop(0.5, 1, 10)
                is_stopped = False
                
    else:
        error = 0
        centroid = get_bottom_right_line_center(image)
        if centroid:
            error = horizontal_distance_from_line(centroid, target_slope, target_intercept)
        else: # if centroid not found, use previous error
            pass
        command = Twist()
        command.linear.x = base_speed
        command.angular.z = error * KP
        move_pub.publish(command)

        cv2.circle(image, centroid, 10, 255, -1)
    cv2.imshow("image", image)
    cv2.waitKey(3)

def turn_left(interval):
    command = Twist()
    command.linear.x = 0.2
    command.angular.z = 0.62    
    move_pub.publish(command)
    rospy.sleep(interval)
    slow_stop(0.2,0.5,5)

def turn_right(interval):
    command = Twist()
    command.linear.x = 0.2
    command.angular.z = -0.62    
    move_pub.publish(command)
    rospy.sleep(interval)
    slow_stop(0.2,0.5,5)

def slow_stop(start_speed, interval, decrements):
    speed_decrement = start_speed / decrements
    time_interval = interval / decrements
    command = Twist()
    for _ in range(decrements):
        command.linear.x = start_speed - speed_decrement
        move_pub.publish(command)
        rospy.sleep(time_interval)
    command = Twist()
    move_pub.publish(command)

def callback(detection_data):
    global redline_detected
    if detection_data.data:
        redline_detected = True
    else:
        redline_detected = False

if __name__ == "__main__":
    redline_detected = False
    bridge = CvBridge()
    base_speed = 0.20
    KP = 0.015
    PID = PID_Control(0.004, 0.000, 0.000, 10, 0.5)
    target_slope, target_intercept = 0.6228, -44

    rospy.init_node("lane_following", anonymous=True)
    move_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
    rospy.sleep(0.5)
    turn_left(3)
    image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, pid_steering, queue_size=1)
    detection_sub = rospy.Subscriber("/redline", Bool, callback, queue_size=1)
    
    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        rate.sleep()