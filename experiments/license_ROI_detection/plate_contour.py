import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import imutils


def proccess_image(data):
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
    cv_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rect_contours(cv_image)

def rect_contours(img):
    ''':param: gray image'''
    
    
    edged = cv2.Canny(gray, 30, 200) 
    cv2.imshow('a', edged)
    cv2.waitKey(0)
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:30]
    screenCnt = None
    
    screenCnts = []
    for c in contours:
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
        if len(approx) == 4:
            screenCnts.append(approx)

    cv2.drawContours(img, screenCnts, -1, (0, 0, 255), 3)

    cv2.imshow('a', img)
    cv2.waitKey(0)
    

if __name__ == '__main__':
    img = cv2.imread('./../test_images/closeup.png')
    
    rect_contours(img)
    quit()

    rospy.init_node('contour', anonymous=True)
    bridge = CvBridge()
    rospy.Subscriber("/R1/pi_camera/image_raw", Image, proccess_image)
    rospy.spin()