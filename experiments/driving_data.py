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
min_period = 0.2
path = "/home/fizzer/Desktop/enph353DrivingLicenceExtraction/experiments/data/"
last_time = datetime.now()
csv_file, csv_writer = None, None
origins, x, y = [], [], []
def data_collection(data):
    global last_time
    time_diff = (datetime.now() - last_time).total_seconds()
    if time_diff < min_period:
        return None
    pose = data.pose.pose
    x_origin, y_origin = -pose.position.y, pose.position.x
    
    quaternion = orientation_to_quaternion(data.pose.pose.orientation)
    roll, pitch, yaw = euler_from_quaternion(quaternion)
    twist = data.twist.twist
    x_velocity, y_velocity = -twist.linear.x*np.sin(yaw), twist.linear.x*np.cos(yaw)

    csv_writer.writerow([x_origin, y_origin, x_velocity, y_velocity])
    last_time = datetime.now()
    

def orientation_to_quaternion(orientation):
    return (
    orientation.x,
    orientation.y,
    orientation.z,
    orientation.w)
    

def open_csv():
    filename = datetime.now().strftime("%H:%M:%S_%d-%m-%y")
    csv_file = open(path + filename + ".csv", 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["x_origin", "y_origin", "x_velocity", "y_velocity"])
    return csv_file, csv_writer

if __name__ == '__main__':
    csv_file, csv_writer = open_csv()
    rospy.init_node('line_follower', anonymous=True)
    rospy.Subscriber("/R1/odom", Odometry, data_collection, queue_size=1)
    rospy.spin()