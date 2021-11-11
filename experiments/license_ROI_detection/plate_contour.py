import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import imutils
from hsv_threshold import hsv_threshold as hsv
import numpy as np
from operator import itemgetter

def rect_contours(img, orig_img):
    ''':param: gray image'''
    orig_img = orig_img.copy()
    def approx_rect(c):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        return approx
    
    edged = cv2.Canny(img, 30, 200) 
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Draw all contours for testing
    # cnt_img = cv2.drawContours(orig_img, contours, -1, (0, 0, 255), 3)
    # cv2.imshow('test', cnt_img)
    # cv2.waitKey(0)

    contours = sorted(contours, key=cv2.contourArea, reverse = True)[:5]

    min_rect_area = 2500
    screenCnts = np.array(list(
        filter(lambda c: cv2.contourArea(c) > min_rect_area,
        filter(lambda x: len(x)==4, 
        map(approx_rect, 
        contours)))))
    allPts = screenCnts.reshape(-1, screenCnts.shape[-1]) if screenCnts.size else screenCnts
    allPts = sorted(allPts, key=itemgetter(1))
    rectPts = np.int32(allPts[:4:2]+allPts[-4::2])
    linePts = rectPts.reshape((-1, 1, 2))

    # Print contour areas for testing
    # contour_areas = list(map(cv2.contourArea, screenCnts))
    # print(contour_areas)

    # Old code
    # screenCnts = []
    # for c in contours:
        
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    #     if len(approx) == 4:
    #         screenCnts.append(approx)
    #         print(approx)

    # Drawing Contours
    cv2.drawContours(orig_img, screenCnts, -1, (0, 0, 255), 3)
    cv2.polylines(orig_img, linePts, True, (0, 255 ,0), 3)
    return orig_img, rectPts
    
def warp_rect(img, in_pts, out_pts, size):
    '''in_pts and out_pts must be of type np.float32 np.int does not work'''
    # To match all points by sorting cartesianly
    inpts, outpts = np.float32(sorted(in_pts.tolist())), np.float32(sorted(out_pts.tolist()))
    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(inpts, outpts)

    # Apply the perspective transformation to the image
    dest = cv2.warpPerspective(img,M,size)
    return dest

if __name__ == '__main__':
    img = cv2.imread('./../test_images/closeup.png')
    thresh = hsv(img)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = rect_contours(thresh, img)
    cv2.imshow('a', img)
    cv2.waitKey(0)