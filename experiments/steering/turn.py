import cv2
import rospy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from green_line_detect import detect_gl
from lane_detection import slope
import math
import numpy as np

class HardTurner:
    def __init__(self):
        self.move_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
        self.aligning = False
        self.cv_bridge = CvBridge()
        self.last_line = 1.0

    def left_turn(self):
        command = Twist()
        command.linear.x = 0.22
        command.angular.z = 0.8  
        self.move_pub.publish(command)
        rospy.sleep(2.2)
        self.stop()

    def stop(self):
        self.move_pub.publish(Twist())

    def right_turn(self):
        command = Twist()
        command.linear.x = 0.185
        command.angular.z = -0.62
        self.move_pub.publish(command)
        rospy.sleep(3)
        self.stop()

    def straight(self, dur):
        command = Twist()
        command.linear.x = 0.15
        self.move_pub.publish(command)
        rospy.sleep(dur)
        self.stop()

    def align(self):
        def aligning(data):
            img = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            line, threshed = detect_gl(img)
            h = threshed.shape[0]
            cv2.imshow('testing', threshed)
            cv2.waitKey(1)
            print(line)
            if not len(line):
                return
            s = slope([line])
            
            print(s)
            command = Twist()
            if np.isclose(0.0, s, atol=0.01):
                mean_y = np.mean([line[1], line[3]])
                if mean_y < 9*h//10:
                    command.linear.x = 0.02
                elif np.isclose(self.last_line, s):
                    self.image_sub.unregister()
                    self.aligning = False
                    print('aligned')
            elif s > 0:
                command.angular.z = -0.05
            elif s < 0:
                command.angular.z = 0.05
            self.last_line  = s
            self.move_pub.publish(command)
        
        self.aligning = True
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, aligning, queue_size=1)
        while self.aligning:
            rospy.sleep(1) 

if __name__ == '__main__':
    rospy.init_node("turnTesting", anonymous=True)
    ht = HardTurner()
    rospy.sleep(0.5)
    ht.align()
    ht.left_turn()
    ht.right_turn()
    ht.straight(2.8)
    ht.align()
    ht.right_turn()
    ht.straight(2.6)
    ht.align()
    ht.right_turn()
    ht.left_turn()
    ht.straight(0.1)
    ht.right_turn()
    # ht.left_turn()
    # left_turn(move_pub)