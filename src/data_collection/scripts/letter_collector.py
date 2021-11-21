import json
from ldriver.licence.detection import LicencePlate
from ldriver.licence.ocr import LicenceOCR
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy
from pathlib2 import Path
import os
import logging
import glob
import re

class ImgCollector:
    data_dir = Path('plate_data')
    def __init__(self):
        if not os.path.exists(str(self.data_dir)):
            os.makedirs(str(self.data_dir))
        l = max([int(re.findall(r'\d+', imgf)[0]) for imgf in glob.glob(str(self.data_dir/'*.png'))])
        self.cur_img_id = int(l)+1
        print('started collecting images at index {}'.format(self.cur_img_id))
        self.prev = None
        self.best_of_group = None

    def proccess_image(self, data):
        cv_img = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        lp = LicencePlate(cv_img)
        if lp.valid:
            if not self.prev:
                self.prev = lp
            if lp == self.prev:
                self.best_of_group = lp if lp.blur > self.prev.blur else self.prev
            else:
                letters = LicenceOCR.process_letters(self.best_of_group.letters)
                for l in letters[1:]:
                    img_f = str(self.data_dir/'{}.png'.format(self.cur_img_id))
                    cv2.imwrite('./'+img_f, l)
                    self.cur_img_id += 1
                    print('saved {}'.format(img_f))
            self.prev = lp


if __name__ == '__main__':
    logging.getLogger("").setLevel(logging.DEBUG)
    rospy.init_node('licensedriver', anonymous=True)
    bridge = CvBridge()
    collector = ImgCollector()
    rospy.Subscriber("/R1/pi_camera/image_raw", Image, collector.proccess_image)
    rospy.spin()