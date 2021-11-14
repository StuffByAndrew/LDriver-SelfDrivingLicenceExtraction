import os
import cv2
import numpy as np
import time

def slope(line):
    x1, y1, x2, y2 = line[0]
    return float(y2-y1)/(x2-x1)

def mask(image, bottom_percent=1):
    top_percent = 1 - bottom_percent
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (0,int(top_percent*image.shape[0])), (image.shape[1], image.shape[0]), 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)

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
    if lines is None:
        return lines
    return filter(lambda line: abs(slope(line)) >= min_slope, lines)

def plot_lines(image, lines, color):
    if not lines:
        return image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), color, 3)

def plot_individual_lines(image, lines, color):
    if not lines:
        return image
    for line in lines:
        print(slope(line))
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), color, 3)
        cv2.imshow("Image", image)
        cv2.waitKey(0)


def sobel(image, x, y, kernel_size):
    return cv2.Sobel(image,cv2.CV_64F,x,y,ksize=kernel_size)  # y

def detect_lines(image):
    line_image = grayscale(image)
    line_image = blur(line_image, kernel_size=1)
    _, line_image = threshold(line_image, lower_threshold=240, upper_threshold=255)
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

def sobel_lines(image):
    line_image = mask(image, 0.3)
    cv2.imshow("Image", line_image)
    cv2.waitKey(0)
    line_image = grayscale(line_image)
    cv2.imshow("Image", line_image)
    cv2.waitKey(0)
    line_image = blur(line_image, kernel_size=15)
    cv2.imshow("Image", line_image)
    cv2.waitKey(0)
    ret, line_image = threshold(line_image, lower_threshold=240, upper_threshold=255)
    cv2.imshow("Image", line_image)
    cv2.waitKey(0)
    line_image = edge_detection(line_image, lower_threshold=30, upper_threshold=130)
    cv2.imshow("Image", line_image)
    cv2.waitKey(0)
    line_image = sobel(line_image, x=0, y=1, kernel_size=5)
    cv2.imshow("Image", line_image)
    cv2.waitKey(0)
    lines = line_detection(line_image.astype(np.uint8), point_threshold=5, min_line_length=100, max_line_gap=5)
    return filter_lines(lines, 0.2)

def split_lines(lines):
    if not lines:
        return None
    left_side, right_side = [], []
    for line in lines:
        if slope(line) > 0:
            right_side.append(line)
        else:
            left_side.append(line)
    return left_side, right_side

def plot_line_centers(image, lines):
    if not lines:
        return image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        center_x = (x2 + x1)//2
        center_y = (y2 + y1)//2
        image = cv2.circle(image, (center_x, center_y), 20, (255,0,0), -1)
    return image

def plot_center(image, lines):
    if not lines: 
        return image
    sides = split_lines(lines)
    if not sides[0] or not sides[1]:
        return image
    averages = []
    for side in sides:
        average_x, average_y, count = 0, 0, 0
        for line in side:
            x1, y1, x2, y2 = line[0]
            average_x += (x2 + x1)/2.0
            average_y += (y2 + y1)/2.0
            count += 1
        averages.append([int(average_x/count), int(average_y/count)])
    center_x = (averages[0][0] + averages[1][0])//2
    center_y = (averages[1][1] + averages[1][1])//2
    return cv2.circle(image, (center_x, center_y), 20, (255,0,0), -1)

if __name__ == "__main__":
    folder = "experiments\\test_images\\"
    for image_file in sorted(os.listdir(folder)):
        print(image_file)
        image = cv2.imread(folder + image_file, cv2.IMREAD_COLOR)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        lines = sobel_lines(image)
        image = plot_center(image, lines)
        # lines = detect_lines(image)
        #plot_lines(image, lines, color=(150,100,250))
        cv2.imshow("Image", image)
        cv2.waitKey(0)


