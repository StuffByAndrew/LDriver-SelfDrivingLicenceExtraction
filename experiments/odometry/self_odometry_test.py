import os
import csv
import time
import rospy
import numpy as np
import message_filters
from datetime import datetime
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist

# Pose/Twist
# position/linear: 
#   x: x
#   y: y
#   z: z
# orientation/angular: 
#   x: x
#   y: y
#   z: z
#   w: w

path = "/home/fizzer/Desktop/enph353DrivingLicenceExtraction/experiments/data/"
csv_file, csv_writer = None, None

class CurrentPosition():
    def __init__(self):
        self.yaw = 0 # radians
        self.omega = 0 # radians per secodn
        self.x, self.y = 0, 0 # meters
        self.x_vel, self.y_vel # meters per second
    
    def set_velocities(self, linear_x, omega):
        self.omega = omega
        self.x = -linear_x*np.sin(self.yaw)
        self.y = linear_x*np.cos(self.yaw)
    
    def time_step(self, interval):
        self.x += self.x_vel * interval
        self.y += self.y_vel * interval
        self.yaw += self.omega * interval
    
    def get_odometry(self):
        return [self.x, self.y, self.x_vel, self.y_vel]

def data_collection(data):
    global last_updated
    global prev_time
    cur_time = rospy.get_time()
    current_position.time_step(cur_time - prev_time)
    if cur_time - last_updated > min_interval:
        csv_writer.writerow(current_position.get_odometry())
        last_updated = cur_time
    current_position.set_velocities(data.linear.x, data.angular.z)
    prev_time = cur_time

def open_csv():
    filename = datetime.now().strftime("%H:%M:%S_%d-%m-%y")
    csv_file = open(path + filename + ".csv", 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["x_origin", "y_origin", "x_velocity", "y_velocity"])
    return csv_file, csv_writer

if __name__ == '__main__':
    current_position = CurrentPosition()
    min_interval = 0.2 # minimum required time between data entries
    last_updated = 0
    prev_time = 0
    csv_file, csv_writer = open_csv()
    current_position = CurrentPosition()
    rospy.init_node('line_follower', anonymous=True)
    rospy.Subscriber("/R1/cmd_vel", Twist, data_collection, queue_size=1)
    rospy.spin()