#!/usr/bin/env python2
# import licence.detection as ld
import cv2
from ldriver.steering.pid_controller import pid_steering, PID_Control
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from ldriver.steering.lane_detection import get_bottom_left_line_center, get_bottom_right_line_center
import rospy

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

counter = 0

def pid_steering(image_data):
    global counter
    try:
        image = bridge.imgmsg_to_cv2(image_data, "bgr8")
    except CvBridgeError as e:
        print(e)
    # image_width = image.shape[1]
    # try:
    #     # licence_img = find_licence(image)
    #     # cv2.imshow("licence", licence_img)
    #     # cv2.waitKey(1)
    #     pass
    # except:
    #     pass
    if counter <= 50:
        centroid = get_bottom_left_line_center(image)
        if centroid:
            PID.calculate_PID(192, centroid[0])
        else: # if centroid not found, use previous error
            PID.calculate_PID(192, PID.error)
    else:
        centroid = get_bottom_right_line_center(image)
        if centroid:
            PID.calculate_PID(target, centroid[0])
        else: # if centroid not found, use previous error
            PID.calculate_PID(target, PID.error)
    cv2.circle(image, centroid, 10, 255, -1)
    cv2.imshow("image", image)
    cv2.waitKey(2)
    command = Twist()
    command.linear.x = base_speed
    command.angular.z = PID.get_error()
    move_pub.publish(command)
    counter += 1

if __name__ == '__main__':
    start_msg, end_msg = String(), String()
    start_msg.data = "A&B,password,0,AAAA"
    end_msg.data = "A&B,password,-1,AAAA"
    bridge = CvBridge()
    base_speed = 0.12
    PID = PID_Control(0.004, 0.000, 0.000, 10, 0.5)
    target = 1088
    counter = 0

    rospy.init_node("lane_following", anonymous=True)
    move_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
    image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, pid_steering, queue_size=1)

    timer_pub = rospy.Publisher("/license_plate", String, queue_size=1)
    rospy.sleep(0.5)
    
    # Start timer
    timer_pub.publish(start_msg)
    start_time = rospy.get_time()
    print("Started Timer")
    
    rate = rospy.Rate(0.5)
    while (rospy.get_time() - start_time) < 60:
        rate.sleep()
    timer_pub.publish(end_msg)