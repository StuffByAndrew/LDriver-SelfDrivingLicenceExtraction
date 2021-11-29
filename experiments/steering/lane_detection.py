import os
import cv2
import numpy as np

def slope(line):
    """ Calculates the slope of a line
    
    Args:
        line (numpy.ndarray): line represented by two points
    Returns:
        float: slope of line
    """
    x1, y1, x2, y2 = line[0]
    return float(y2-y1)/(x2-x1)

def slope_point_form(line):
    x1, y1, x2, y2 = line[0]
    m = float(y2-y1)/(x2-x1)
    b = y1 - m*x1 
    return m, b  

def horizontal_distance_from_line(point, slope, y_intercept):
    x1 = point[0]
    x2 = (point[1] - y_intercept)/slope
    return x2-x1

def filter_lines(lines, min_slope):
    """ Returns numpy.ndarray of lines with abs(slopes) greater than or equal to min_slope
    
    Args:
        lines (numpy.ndarray): input lines
        min_slope (float): minimum slope
    Returns
        numpy.ndarray: filtered lines
    """
    if lines is None: return None
    return filter(lambda line: abs(slope(line)) >= min_slope, lines)

def split_lines_slope(lines):
    """ Split lines based on positive or negative slopes 
    (Positive slopes tend to represent left roadlines, negative -> right roadlines)

    Args: 
        lines (numpy.ndarray): input lines
    Returns:
        list: list of lines with positive slope
        list: list of lines with negative slope
    """
    if not lines:
        return None
    left_side, right_side = [], []
    for line in lines:
        if slope(line) > 0:
            right_side.append(line)
        else:
            left_side.append(line)
    return left_side, right_side

def split_lines_centroid(centroid, lines):
    """ Split lines based on left or right of centroid

    Args: 
        lines (numpy.ndarray): input lines
    Returns:
        list: list of lines left of centroid
        list: list of lines right of centroid
    """
    if not lines:
        return None
    left_side, right_side = [], []
    for line in lines:
        # TODO: split lines better
        if line[0][0] < centroid[0] or line[1][0] < centroid[0]:
            left_side.append(line)
        elif line[0][0] > centroid[0] or line[1][0] > centroid[1]:
            right_side.append(line)
        else:
            pass
    return left_side, right_side
    
def mask_rectangle(image, left=0, top=0, right=0, bottom=0):
    """ Returns an image with left, top, right, bottom percent of the image masked out and the remaining 
    portion untouched.
    
    Args:
        image (numpy.ndarray): input image
        left, top, right, bottom (float, optional): percent of image (from the bottom) left unmasked represented
        as a decimal between 0 and 1 inclusive. Defaults to 1
    Returns:
        numpy.ndarray: masked output image
    """
    if not 0 <= left <= 1 and 0 <= top <= 1 and 0 <= right <= 1 and 0 <= bottom <= 1:
        raise ValueError("left, top, right, nd bottom must be floats between 0 and 1 inclusive.")
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (int(left*image.shape[1]), int(top*image.shape[0])), (int((1 - right)*image.shape[1]), int((1- bottom)*image.shape[0])), 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)

def hsv_threshold(img, lh=0, uh=0, ls=0, us=0, lv=0, uv=0):
    """ Thresholds an image based on hsv 
    
    Args:
        lh, uh, ls, us, lv, uv (int): lower and upper hue, saturation, and value thresholds
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.medianBlur(hsv,5)
    uh = uh
    us = us
    uv = uv
    lh = lh
    ls = ls
    lv = lv
    lower_hsv = np.array([lh,ls,lv])
    upper_hsv = np.array([uh,us,uv])

    return cv2.inRange(hsv, lower_hsv, upper_hsv)

def get_roadlines(image):
    """ Obtains the roadlines from the image

    Args:
        image (numpy.ndarray): input image
    Returns:
        numpy.ndarray: ndarray of roadlines
    """
    line_image = mask_rectangle(image, top=0.7)
    line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    line_image = cv2.GaussianBlur(line_image, ksize=(15, 15), sigmaX=0)
    _, line_image = cv2.threshold(line_image, thresh=240, maxval=255, type=cv2.THRESH_BINARY)
    line_image = cv2.Canny(line_image, threshold1=30, threshold2=130)
    line_image = cv2.Sobel(line_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    lines = cv2.HoughLinesP(line_image.astype(np.uint8), rho=1, theta=np.pi/180, threshold=5, minLineLength=5, maxLineGap=5)
    return filter_lines(lines, 0.2)

def get_lines_center(lines):
    """ Obtains the center of the roadlines when both sides are detected, otherwise outputs None
    
    Args: 
        image (np.ndarray): input lines
    Returns:
        tuple: x-y coordinates of the center of the lines
    """
    average_x, average_y, count = 0, 0, 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        average_x += x2 + x1
        average_y += y2 + y1
        count += 2
    return (average_x//count, average_y//count)

def get_roadlines_center(image):
    """ Obtains the center of the roadlines when both sides are detected, otherwise outputs None
    
    Args: 
        image (np.ndarray): input image
    Returns:
        tuple: x-y coordinates of the center of the roadlines. None if both sides were not detected
    """
    lines = get_roadlines(image)
    if not lines: return None
    sides = split_lines_slope(lines)
    if not sides[0] or not sides[1]: return None
    averages = []
    for side in sides:
        averages.append(get_lines_center(side))
    center_x = (averages[0][0] + averages[1][0])//2
    center_y = (averages[1][1] + averages[1][1])//2
    return (center_x, center_y)

def find_center(image):
    """ Finds the centroid of an object

    Args:
        image (np.ndarray): input image
    Returns:
        tuple: x-y coordinates of the centroid
    """
    M = cv2.moments(image)
    try:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    except ZeroDivisionError as e:
        return None
    return (cx, cy)

def dilate_erode(img, dilation_kernel_size=0, erosion_kernel_size=0):
    """ Denoise salt and pepper noise in an image through dilation and erosion

    Args:
        img (numpy.ndarray): binary image matrix

    Returns:
        numpy.ndarray: denoised binary image
    """
    dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    img = cv2.dilate(img, dilation_kernel, iterations=1)
    img = cv2.erode(img, erosion_kernel, iterations=1)
    return img

def floodfill(input_image):
    image = input_image.copy()
    height, width = image.shape[:2]
    mask = np.zeros((height+2, width+2), np.uint8)
    image = image.astype("uint8")
    cv2.floodFill(image, mask, (0,0), 255)
    negative = cv2.bitwise_not(image)
    return input_image | negative

def get_road_mask(input_image):
    image = cv2.medianBlur(input_image, 5)
    image = hsv_threshold(image, lh=0, ls=0, lv=75, uh=0, us=0, uv=95)
    image = dilate_erode(image, 50, 50)
    image = dilate_erode(image, 50, 50)
    return floodfill(image)

def mask_road(input_image):
    mask = get_road_mask(input_image)
    return cv2.bitwise_and(input_image, input_image, mask=mask)

def get_roadcolor_center(image):
    center_image = mask_rectangle(image, top=0.7)
    center_image = mask_road(center_image)
    center_image = hsv_threshold(center_image, lv=75, uv=95)
    _, center_image = cv2.threshold(center_image, thresh=240, maxval=255, type=cv2.THRESH_BINARY)
    return find_center(center_image)

def get_bottom_right_line_center(image):
    line_image = mask_rectangle(image, left=0.5, top=0.5)
    line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    _, line_image = cv2.threshold(line_image, thresh=240, maxval=255, type=cv2.THRESH_BINARY)
    return find_center(line_image)

def get_bottom_left_line_center(image):
    line_image = mask_rectangle(image, right=0.7, top=0.7)
    line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    _, line_image = cv2.threshold(line_image, thresh=240, maxval=255, type=cv2.THRESH_BINARY)
    return find_center(line_image)

def get_right_lines_center(image):
    lines = get_roadlines(image)
    _, right_side = split_lines_slope(lines)
    return get_lines_center(right_side)

# if __name__ == "__main__":
#     folder = "experiments/test_images/"
#     for image_file in sorted(os.listdir(folder)):
#         print(image_file)
#         image = cv2.imread(folder + image_file, cv2.IMREAD_COLOR)
#         center = get_bottom_right_line_center(image)
#         cv2.circle(image, center, 10, 255, -1)
#         center = get_bottom_left_line_center(image)
#         cv2.circle(image, center, 10, 255, -1)
#         cv2.imshow("Image", image)
#         cv2.waitKey(0)
if __name__ == "__main__":
    image = cv2.imread("img_cmd_data/13.png")
    cv2.imshow("a", image)
    cv2.waitKey(0)
    image = mask_rectangle(image, left=0.5)
    lines = get_roadlines(image)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (255), 3)
        print(slope_point_form(line))
        cv2.imshow("a", image)
        cv2.waitKey(0)  

#(0.6226415094339622, -45.358490566037744)
#(0.6228070175438597, -43.07017543859649)