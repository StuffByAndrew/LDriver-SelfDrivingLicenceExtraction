import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def proccess_image(data):
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
    rect_contours(cv_image)

def rect_contours(img):
    ''':param: img image'''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    cnts= cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    print(cnts)
    cv2.drawContours(img, cnts, -1, (0,255,0), 3)
    cv2.imshow('a', img)
    cv2.waitKey(0)
    return
    
    cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

    # loop over our contours
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            cv2.drawContours(img, [approx], -1, (0,255,0), 3)
    
    cv2.imshow('a', img)
    cv2.waitKey(0)

    

if __name__ == '__main__':
    img = cv2.imread('test_license.png')
    
    rect_contours(img)
    quit()

    rospy.init_node('contour', anonymous=True)
    bridge = CvBridge()
    rospy.Subscriber("/R1/pi_camera/image_raw", Image, proccess_image)
    rospy.spin()