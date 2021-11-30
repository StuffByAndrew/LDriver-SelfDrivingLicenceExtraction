import numpy as np
import cv2

def detect_gl(img):
    img = cv2.medianBlur(img, 5)
    h,w = img.shape[:2]
    img = img[3*h//5:,w//3:2*w//3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    u = np.array([90, 80, 129])
    l = np.array([55, 4, 82])
    thresh = cv2.inRange(img, l, u)
    thresh = cv2.Canny(thresh, 50, 150)
    linesP = cv2.HoughLinesP(thresh.astype(np.uint8), rho=1, theta=np.pi/180, threshold=15, minLineLength=200, maxLineGap=50)
    
    longest = []
    if linesP is not None:
        linesP = np.array(linesP).reshape(-1, 4) 
        lengths = np.linalg.norm(np.vstack((linesP[:,2]-linesP[:,0], linesP[:,3]-linesP[:,1])).T, axis=1)
        longest = linesP[np.argmax(lengths)]
    return longest, thresh