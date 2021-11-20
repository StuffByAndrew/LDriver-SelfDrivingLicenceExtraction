import json
from ldriver.licence.detection import LicencePlate
from ldriver.licence.ocr import LicenceOCR
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy
from pathlib2 import Path
import os

class ImgCollector:
    data_dir = Path('plate_data')
    def __init__(self):
        if not os.path.exists(str(self.data_dir)):
            os.makedirs(str(self.data_dir))
        self.cur_img_id = 0

    def proccess_image(self, data):
        cv_img = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        try:
            lp = LicencePlate(cv_img)
            if lp.valid:
                letters = LicenceOCR.process_letters(lp.letters)
                for l in letters:
                    img_f = str(self.data_dir/'{}.png'.format(self.cur_img_id))
                    cv2.imwrite('./'+img_f, plate)
                self.cur_img_id += 1
                print('saved{}'.format(img_f))
        except:
            pass

if __name__ == '__main__':
    rospy.init_node('licensedriver', anonymous=True)
    bridge = CvBridge()
    collector = ImgCollector()
    rospy.Subscriber("/R1/pi_camera/image_raw", Image, collector.proccess_image)
    rospy.spin()