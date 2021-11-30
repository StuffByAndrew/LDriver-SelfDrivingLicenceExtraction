from ldriver.licence.detection import dilate_erode
from lane_detection import slope
import cv2
import numpy as np

def detect_gl(img):
    img = cv2.medianBlur(img, 5)
    h,w = img.shape[:2]
    img = img[h//2:,w//3:2*w//3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    u = np.array([82, 75, 97])
    l = np.array([56, 27, 84])
    thresh = cv2.inRange(img, l, u)
    thresh = cv2.Canny(thresh, 50, 150)
    linesP = cv2.HoughLinesP(thresh.astype(np.uint8), rho=1, theta=np.pi/180, threshold=15, minLineLength=200, maxLineGap=50)
    
    longest = []
    if linesP is not None:
        linesP = np.array(linesP).reshape(-1, 4) 
        lengths = np.linalg.norm(np.vstack((linesP[:,2]-linesP[:,0], linesP[:,3]-linesP[:,1])).T, axis=1)
        longest = linesP[np.argmax(lengths)]
    return longest, thresh

if __name__ == '__main__':
    import glob
    # orig_img = cv2.imread('./experiments/test_images/74.png')
    for imgf in glob.glob('img_cmd_data/*.png'):
        img = cv2.imread(imgf)
        longest, thresh = detect_gl(img)
        print(longest)
        cv2.imshow('test', thresh)
        cv2.waitKey(0)