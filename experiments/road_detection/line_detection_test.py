import cv2
import numpy as np
import time

def slope_intercept_form(lines):
    line_data = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = float(y1-y2)/(x1-x2)
        intercept = y1 - slope*x1
        length = np.sqrt((y2-y1)**2+(x2-x1)**2)
        line_data.append((line[0], slope, intercept, length))
    return line_data
            
kernel_size = 5
image = cv2.imread("test_images/P2_mid.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
ret, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)
edge = cv2.Canny(thresh, 50, 130)
cv2.imshow("Image", edge)
cv2.waitKey(0)
lines = cv2.HoughLinesP(edge, rho=1, theta=np.pi/180, threshold=20, minLineLength=250, maxLineGap=2000)
try:
    if False:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (20, 255, 255), 3)
    lines = slope_intercept_form(lines)
    for line in lines:
        slope = line[1]
        if abs(slope) < 0.3:
            continue
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
except:
    print("No Lines")

