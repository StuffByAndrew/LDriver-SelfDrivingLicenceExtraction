#!/usr/bin/env python2
import cv2
import time
import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Int16
from cv_bridge import CvBridge, CvBridgeError
from ldriver.steering.lane_detection import right_roadline_center, horizontal_distance_from_line
from ldriver.steering.pedestrian_detection import pedestrian_crossing
from ldriver.steering.road_detection import detect_gl
from ldriver.steering.lane_detection import slope
bridge = CvBridge()

class Steering_Control:
    def __init__(self, base_speed, KP, target_line, move_pub):
        self.base_speed = base_speed
        self.KP = KP
        self.target_line = target_line
        self.move_pub = move_pub
        
    def get_error(self, point):
        error = horizontal_distance_from_line(point, *self.target_line)
        return self.KP*error
    
    def auto_steer(self, image):
        centroid = right_roadline_center(image)
        error = self.get_error(centroid)
        #--------------------------
        cv2.circle(image, centroid, 10, 255, -1)
        cv2.imshow("image", image)
        cv2.waitKey(3)
        #--------------------------
        command = Twist()
        command.linear.x = self.base_speed
        command.angular.z = error
        self.move_pub.publish(command)
    
    def move_forwards(self, duration):
        command = Twist()
        command.linear.x = 0.5
        self.move_pub.publish(command)
        rospy.sleep(duration)
    
    def slow_stop(self, start_speed, end_speed, interval, decrements):
        speed_decrement = (start_speed - end_speed) / decrements
        time_interval = interval / decrements
        command = Twist()
        for _ in range(decrements):
            command.linear.x = start_speed - speed_decrement
            self.move_pub.publish(command)
            rospy.sleep(time_interval)
    
    def turn_left(self, interval):
        command = Twist()
        command.linear.x = 0.2
        command.angular.z = 0.62    
        self.move_pub.publish(command)
        rospy.sleep(interval)
        self.slow_stop(0.2,0,1,5)
    
    def stop(self):
        self.move_pub.publish(Twist())

class Pedestrian_Detection:
    def __init__(self, threshold, history_length):
        self.threshold = threshold
        self.history = [0 for i in range(history_length)]
        self.previous_image = None
        self.pedestrian_was_crossing = False

    def robot_should_cross(self, current_image):
        running_average, self.previous_image = pedestrian_crossing(current_image, self.previous_image, self.history)
        pedestrian_currently_crossing = running_average > self.threshold
        if not pedestrian_currently_crossing and self.pedestrian_was_crossing:
            self.pedestrian_was_crossing, self.previous_image = False, None
            return True
        else:
            self.pedestrian_was_crossing = pedestrian_currently_crossing
            return False

class HardTurner:
    def __init__(self, move_pub):
        self.move_pub = move_pub
        self.aligning = False
        self.cv_bridge = CvBridge()
        self.last_line = 1.0
        self.lost_dur = 0

    def left_turn(self):
        command = Twist()
        command.linear.x = 0.35 # 0.22
        command.angular.z = 1.05 # 0.8  
        self.move_pub.publish(command)
        rospy.sleep(1.5)
        self.stop()

    def stop(self):
        self.move_pub.publish(Twist())

    def right_turn(self):
        command = Twist()
        command.linear.x = 0.35 # 0.185
        command.angular.z = -1.06 # -0.62
        self.move_pub.publish(command)
        rospy.sleep(1.5)
        self.stop()

    def straight(self, dur):
        command = Twist()
        command.linear.x = 0.3
        self.move_pub.publish(command)
        rospy.sleep(dur)
        self.stop()

    def back(self, dur, speed=0.3):
        command = Twist()
        command.linear.x = -speed
        self.move_pub.publish(command)
        rospy.sleep(dur)
        self.stop()

    def align(self):
        def aligning(data):
            img = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            line, threshed = detect_gl(img)
            h = threshed.shape[0]
            # cv2.imshow('testing', threshed)
            # cv2.waitKey(1)
            print(line)
            if not len(line):
                self.lost_dur += 1
                if self.lost_dur > 10:
                    self.back(0.1, speed=0.08)
                return
            s = slope([line])
            self.lost_dur = 0 
            print(s)
            command = Twist()
            if np.isclose(0.0, s, atol=0.01):
                mean_y = np.mean([line[1], line[3]])
                if mean_y < 9*h//10:
                    command.linear.x = 0.08
                elif np.isclose(self.last_line, s):
                    self.image_sub.unregister()
                    self.aligning = False
                    print('aligned')
            elif s > 0:
                command.angular.z = -0.07
            elif s < 0:
                command.angular.z = 0.07
            self.last_line  = s
            self.move_pub.publish(command)
        
        self.aligning = True
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw2", Image, aligning, queue_size=1)
        while self.aligning:
            rospy.sleep(1)

class Detection:
    def __init__(self):
        self._redline_detected = None
    
    @property
    def detected(self):
        return self._redline_detected
    
    @detected.setter
    def detected(self, update):
        self._redline_detected = update

def update_redline(detection):
    if detection.data:
        Redline.detected = True
    else:
        Redline.detected = False

def update_greenline(detection):
    if detection.data:  
        Greenline.detected = True
    else:
        Greenline.detected = False

def update_license_number(detection):
    LicenseNumber.detected = detection.data

def update_image(image):
    Current_Image.detected = image

def autopilot(image_data):
    try:
        image = bridge.imgmsg_to_cv2(image_data, "bgr8")
    except CvBridgeError as e:
        print(e)
    #-------
    update_image(image)
    if Redline.detected:
        Steering.stop()
        if Pedestrian.robot_should_cross(image):
            Steering.move_forwards(1.25)
    elif LicenseNumber.detected == 1 and Greenline.detected:
        Steering.stop()
        print('straight')
        ht.straight(0.2)
        print('align')
        ht.align()
        print('straight')
        ht.straight(0.2)
        print('left')
        ht.left_turn()
        ht.back(0.4)
        ht.align()
        ht.left_turn()
        ht.back(0.1)
        ht.straight(0.35)
        ht.right_turn()
        ht.straight(1.5)
        ht.right_turn()
        ht.straight(1.5)
        ht.align()
        ht.right_turn()
        ht.left_turn()
        ht.back(0.2)
        ht.left_turn()
        ht.back(1)
    else:
        Steering.auto_steer(image)

if __name__ == "__main__":
    rospy.init_node("autopilot", anonymous=True)
    move_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
    ht = HardTurner(move_pub)
    Steering = Steering_Control(0.20, 0.015, (0.6228, -44), move_pub)
    Pedestrian = Pedestrian_Detection(200, 5)
    Redline = Detection()
    Greenline = Detection()
    LicenseNumber = Detection()
    Current_Image = Detection()
    
    rospy.sleep(0.5)
    Steering.turn_left(3)
    
    image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, autopilot, queue_size=1)
    redline_sub = rospy.Subscriber("/redline", Bool, update_redline, queue_size=1)
    greenline_sub = rospy.Subscriber("/RoadDetection/greenline", Bool, update_greenline, queue_size=1)
    parking_number_sub = rospy.Subscriber("license_id", Int16, update_license_number, queue_size=1)
    
    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        rate.sleep()