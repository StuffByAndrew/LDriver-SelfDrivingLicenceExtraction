#!/usr/bin/env python2
from ldriver.licence.detection import LicencePlate
import cv2
from matplotlib.pyplot import imshow
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy
from ldriver.licence.ocr import LicenceOCR
import numpy as np
from itertools import starmap
from std_msgs.msg import String, Int16
from scoring import TEAM_NAME, TEAM_PWD

HOR_LINE = '-' * 30

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
        def publish_to_scoring(id):
            pred_str = ''.join(self.best[id]['pred'])
            publish_scoring(plate_id=p_space, plate_num=pred_str[-4:])
        
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
                publish_to_scoring(p_space)
        else:
            self.best[p_space] = {
                'conf': conf,
                'pred': preds
            }
            print('recorded {} at {}'.format(self.best[p_space]['pred'], self.best[p_space]['conf']))
            publish_to_scoring(p_space)
        
        # Publish to Licence ID publisher for turning into the center decision
        lid_pub.publish(Int16(int(p_space)))

        # print horizontal line
        print(HOR_LINE)
    
def publish_scoring(plate_id, plate_num):
    if plate_id == 0:
        return
    scoring_pub.publish(str('{},{},{},{}').format(
        TEAM_NAME,
        TEAM_PWD,
        plate_id,
        plate_num
    ))


if __name__ == '__main__':
    rospy.init_node('licensedriver')
    ld = LicenceDetector()
    locr = LicenceOCR()
    bridge = CvBridge()
    scoring_pub = rospy.Publisher('/license_plate', String, queue_size=1)
    lid_pub = rospy.Publisher('/license_id', Int16, queue_size=1)
    rospy.sleep(1)
    # scoring_pub.publish(str('{},{},0,XR58').format(
    #     TEAM_NAME,
    #     TEAM_PWD
    # ))
    rospy.Subscriber("/R1/pi_camera/image_raw", Image, ld.process_image)
    rospy.spin()