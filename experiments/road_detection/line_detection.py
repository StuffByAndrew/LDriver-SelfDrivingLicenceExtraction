import cv2
import numpy as np
import time

def slope(line):
    x1, y1, x2, y2 = line[0]
    return float(y2-y1)/(x2-x1)

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def threshold(image, lower_threshold, upper_threshold): 
    return cv2.threshold(image, lower_threshold, upper_threshold, cv2.THRESH_BINARY)

def edge_detection(image, lower_threshold, upper_threshold):
    return cv2.Canny(image, lower_threshold, upper_threshold)

def line_detection(image, point_threshold, min_line_length, max_line_gap):
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=point_threshold, 
    minLineLength=min_line_length, maxLineGap=max_line_gap)

def filter_lines(lines, min_slope):
    return filter(lambda line: slope(line) >= min_slope, lines)

def plot_lines(image, lines, color):
    if not lines:
        return image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), color, 3)

def detect_lines(image):
    line_image = grayscale(image)
    line_image = blur(line_image, kernel_size=1)
    ret, line_image = threshold(line_image, lower_threshold=240, upper_threshold=255)
    cv2.imshow("Image", line_image)
    cv2.waitKey(0)
    line_image = edge_detection(line_image, lower_threshold=30, upper_threshold=130)
    cv2.imshow("Image", line_image)
    cv2.waitKey(0)
    try:
        lines = line_detection(line_image, point_threshold=10, min_line_length=10, max_line_gap=20000)
        return filter_lines(lines, 0.3)
    except:
        print("No lines found")
        return None

if __name__ == "__main__":
    image_file = "test_images/P4_close.png"
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    lines = detect_lines(image)
    plot_lines(image, lines, color=(255,255,0))
    cv2.imshow("Image", image)
    cv2.waitKey(0)




   

