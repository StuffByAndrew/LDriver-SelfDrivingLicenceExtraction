import cv2
import numpy as np

def mask_rectangle(input_image, left=0, top=0, right=0, bottom=0):
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
    mask = np.zeros(input_image.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (int(left*input_image.shape[1]), int(top*input_image.shape[0])), (int((1 - right)*input_image.shape[1]), int((1- bottom)*input_image.shape[0])), 255, -1)
    return cv2.bitwise_and(input_image, input_image, mask=mask)

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

def roadline_center(input_image, right):
    """ Obtain centroids of the roadline left or right of robot

    Args: 
        input_image (numpy.ndarray): input image
        right (bool): if True obtains right roadline centroid, 
        else left roadline centroid
    Returns:
        tuple: centroid of right or left roadline
    """
    image = mask_rectangle(input_image, top=0.5)
    if right:
        line_image = mask_rectangle(image, left=0.5)
    else:
        line_image = mask_rectangle(image, right=0.5)
    line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    _, line_image = cv2.threshold(line_image, thresh=240, maxval=255, type=cv2.THRESH_BINARY)
    return find_center(line_image)

def right_roadline_center(input_image):
    """ Obtain centroids of the roadline left or right of robot

    Args: 
        input_image (numpy.ndarray): input image
    Returns:
        tuple: centroid of right roadline
    """
    line_image = mask_rectangle(input_image, top=0.5, left=0.5)
    line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    _, line_image = cv2.threshold(line_image, thresh=240, maxval=255, type=cv2.THRESH_BINARY)
    return find_center(line_image)

def horizontal_distance_from_line(point, slope, y_intercept):
    """ Obtain the horizontal distance from the point to a 
    line defined by slope and y_intercept

    Args: 
        point (tuple): input point
        slope (float): slope of target line
        y_intercept: y-intercept of target line
    Returns:
        float: distance
    """
    point_x = point[0]
    line_x = float(point[1] - y_intercept)/slope
    return line_x - point_x

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

def flip_image(input_image):
    return np.flipud(input_image)
