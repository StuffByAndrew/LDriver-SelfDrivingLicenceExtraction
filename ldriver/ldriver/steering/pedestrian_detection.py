import sys
import cv2
import numpy as np
from ldriver.steering.lane_detection import hsv_threshold, mask_rectangle

def function_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    return helper

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

@function_counter
def pedestrian_crossing(current_image_input, previous_image_output, history):
    """ Compare current image to previous image and determine if pedestrian

    Args:
        current_image_input (numpy.ndarray): current image
        previous_image_output (numpy.ndarray): previous image output
        history (list): list of previous non zero values
    Returns:
        int: running average
    """
    current_image = hsv_threshold(current_image_input, lh=88, ls=40, lv=40, uh=113, us=255, uv=128)
    current_image = cv2.GaussianBlur(current_image, (21,21), 0)
    current_image = dilate_erode(current_image, 0, 10)
    if previous_image_output is None:
        return 0, current_image
    difference = cv2.absdiff(current_image, previous_image_output)
    _, thresh = cv2.threshold(difference,55,255,cv2.THRESH_BINARY)
    cv2.imshow("image", thresh)
    cv2.waitKey(1)
    cv2.imshow("image", current_image_input)
    cv2.waitKey(1)
    history.append(np.count_nonzero(thresh))
    history.pop(0)
    running_average = sum(history) / len(history)
    return running_average, current_image
   
def redline_detected(input_image, bottom_percent=1):
    """Detects redline within the bottom_percent of the image
    
    Args:
        input_image (np.ndarray): input image
        bottom_percent (float): bottom percentage of screen used to check for redline
    Returns:
        bool: True if redline is detected, false otherwise
    """
    image = mask_rectangle(input_image, left=0.2, top=(1-bottom_percent), right=0.2, bottom=0)
    image = hsv_threshold(image, lh=0, ls=10, lv=90, uh=0, us=255, uv=255)
    return np.any(image)