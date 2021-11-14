#!/usr/bin/env python2
import cv_bridge
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rospy
import json
from cv_bridge import CvBridge
from pathlib2 import Path
import os
import cv2

class ImgCmdCollector:
    data_dir = Path('img_cmd_data')
    def __init__(self):
        if not os.path.exists(str(self.data_dir)):
            os.makedirs(str(self.data_dir))
        self.commands = self.load_cmds()
        print(self.commands.keys())
        self.cur_img_id = max(list(map(int,self.commands.keys()))) + 1 if self.commands.keys() else 1
        print('started collecting at index {}'.format(self.cur_img_id))

    def load_cmds(self):
        try:
            with open(str(self.data_dir/'commands.json'), 'w+') as f:
                commands = json.load(f)
                print('loaded commands file')
        except:
            commands = {}
        return commands

    def save_cmds(self, data):
        with open(str(self.data_dir/'commands.json'), 'w+') as f:
            f.write(json.dumps(data, indent=4))

    def callback(self, img, cmd, odom):
        cv_image = bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')

        img_f = str(self.data_dir/'{}.png'.format(self.cur_img_id))
        print('saving {}'.format(img_f))
        cv2.imwrite('./'+img_f, cv_image)
        pose = odom.pose.pose
        twist = odom.twist.twist
        self.commands[self.cur_img_id] = {
            'linear': cmd.linear.x,
            'angular': cmd.angular.z,
            'state': {
                'x': -pose.position.y,
                'y': pose.position.x,
                'linear': twist.linear.x,
                'angular': twist.angular.z
            }
        }
        self.save_cmds(self.commands)
        print('Recorded: {}'.format(self.cur_img_id))
        #TODO: Also try to break this down to L, R, F, None   

        self.cur_img_id += 1

if __name__ == '__main__':
    bridge = CvBridge()
    dc = ImgCmdCollector()
    rospy.init_node('imgcmdcollector', anonymous=True)
    image_sub = message_filters.Subscriber("/R1/pi_camera/image_raw", Image)
    cmd_sub = message_filters.Subscriber("/R1/cmd_vel", Twist)
    odom_sub = message_filters.Subscriber("/R1/odom", Odometry)

    ts = message_filters.ApproximateTimeSynchronizer([image_sub, cmd_sub, odom_sub], 1, 1, allow_headerless=True)
    ts.registerCallback(dc.callback)
    rospy.spin()