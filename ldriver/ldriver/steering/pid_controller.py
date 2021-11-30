from cv_bridge import CvBridge, CvBridgeError
from lane_detection import flip_image, right_roadline_center, horizontal_distance_from_line
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class PID_Controller():
    def __init__(self, basespeed, keep_right, KP, target_line, move_pub):
        self.bridge = CvBridge()
        #-------------------------
        self.basespeed = basespeed
        self.keep_right = keep_right
        self.KP = KP
        self.target_line = target_line
        self.move_pub = move_pub

    def convert_image_data(self, image_data):
        try:
            return self.bridge.imgmsg_to_cv2(image_data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def get_error(self, image):
        centroid = right_roadline_center(image)
        return horizontal_distance_from_line(centroid, *self.target_line)
    
    def steer(self, image_data):
        image = self.convert_image_data(image_data)
        if not self.keep_right:
            image = flip_image(image)
        error = self.KP * self.get_error(image)
        
        command = Twist()
        command.linear.x = self.basespeed
        command.linear.z = error
        self.move_pub.publish()
    
    def follow_left(self, image_data):
        pass