#!/usr/bin/env python2
import rospy
from std_msgs.msg import String

TEAM_NAME = 'Brian_N_Andrew'
TEAM_PWD = 'multi21'
MAX_TIME = 4 # minutes

def format_message(plate_id, plate_str):
    return "{},{},{},{}".format(
        TEAM_NAME,
        TEAM_PWD,
        plate_id,
        plate_str
    )

if __name__ == "__main__":
    start_msg, end_msg = String(), String()
    start_msg.data, end_msg.data = format_message(0,"0000"), format_message(-1,"0000")
    rospy.init_node("scoring", anonymous=True)
    timer_pub = rospy.Publisher("/license_plate", String, queue_size=1)
    rospy.sleep(0.5)
    
    rate = rospy.Rate(0.5)
    timer_pub.publish(start_msg)
    start_time = rospy.get_time()
    rospy.logdebug("Started Timer")
    while (rospy.get_time() - start_time) < 60 * MAX_TIME:
        rate.sleep()
    timer_pub.publish(end_msg)