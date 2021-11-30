#!/usr/bin/env python2
from ldriver.licence.detection import LicencePlate
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy
from ldriver.licence.ocr import LicenceOCR
import numpy as np
from itertools import starmap

HOR_LINE = '-' * 20

class LicenceDetector:
    def __init__(self):
        self.best = {}

    def process_image(self, data):
        cv_img = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        lp = LicencePlate(cv_img) 
        if not lp.valid:
            return

        cv2.imshow('plate', lp.img)
        cv2.waitKey(1)
        preds, conf = locr.read_licence(lp)
        p_space = preds[1]
        
        # Replace old letter predictions if new predictions have a higher confidence
        if p_space in self.best:
            if np.all((conf < self.best[p_space]['conf'])):
                print('less confidence')
            elif preds == self.best[p_space]['pred']:
                print('same prediction')
            else:
                prev = self.best[p_space]
                # Print confidence
                print('comparing {} to {}'.format(prev['pred'], preds))
                print('comparing {} to {}'.format(prev['conf'], conf))
                old = ''.join(prev['pred'])

                # Replace new predictions if confidence is higher
                prev['pred'] = list(starmap(
                    (lambda l,l_old,c,c_old: l if c > c_old else l_old), zip(preds,prev['pred'],conf,prev['conf'])))
                
                # Replace confidence values
                prev['conf'] = np.maximum(prev['conf'], conf)

                # Print predictions
                new = ''.join(prev['pred'])
                print('replaced {} with {}\n{}'.format(old, new, prev['conf']))
        else:
            self.best[p_space] = {
                'conf': conf,
                'pred': preds
            }
            print('recorded {} at {}'.format(self.best[p_space]['pred'], self.best[p_space]['conf']))
        # print horizontal line
        print(HOR_LINE)


if __name__ == '__main__':
    rospy.init_node('licensedriver', anonymous=True)
    ld = LicenceDetector()
    locr = LicenceOCR()
    orig_img = cv2.imread('experiments/test_images/74.png')
    # lp = LicencePlate(orig_img)
    # if lp.valid:
    #     res = locr.read_licence(lp)
    #     print(res)
    #     cv2.imshow('a', lp.img)
    #     cv2.waitKey(0)
    bridge = CvBridge()
    rospy.Subscriber("/R1/pi_camera/image_raw", Image, ld.process_image)
    rospy.spin()