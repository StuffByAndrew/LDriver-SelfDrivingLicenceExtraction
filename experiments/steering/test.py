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
previous_image = None
person_was_crossing = False
history = [0,0,0]
def pid_steering(image_data):
    global previous_image
    global person_currently_crossing
    try:
        image = bridge.imgmsg_to_cv2(image_data, "bgr8")
    except CvBridgeError as e:
        print(e)
    #current_image = cv2.cvtColor(current_image_input, cv2.COLOR_BGR2GRAY)
    current_image = hsv_threshold(image, lh=88, ls=40, lv=40, uh=113, us=255, uv=128)
    current_image = cv2.GaussianBlur(current_image, (21,21), 0)
    current_image = dilate_erode(current_image, 0, 5)

    if previous_image is None:
        previous_image = current_image
        print("Person Not Currently Crossing")
    
    difference = cv2.absdiff(current_image, previous_image)
    _, thresh = cv2.threshold(difference,55,255,cv2.THRESH_BINARY)
    history.append(np.count_nonzero(thresh))
    history.pop(0)
    print(history)
    print(sum(history)/len(history))
    previous_image = current_image
    cv2.imshow("thresh", image)
    cv2.waitKey(1)
    cv2.imshow("thresh", thresh)
    cv2.waitKey(1)

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
    #turn_left(3)
    image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, pid_steering, queue_size=1)
    detection_sub = rospy.Subscriber("/redline", Bool, callback, queue_size=1)
    
    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        rate.sleep()