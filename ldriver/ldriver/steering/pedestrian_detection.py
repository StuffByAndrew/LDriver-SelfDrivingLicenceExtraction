import sys
import cv2
import numpy as np
from ldriver.road.lane_detection import hsv_threshold

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

def pedestrian_crossing(current_image_input, previous_image_output, history):
    current_image = hsv_threshold(current_image_input, lh=88, ls=40, lv=40, uh=113, us=255, uv=128)
    current_image = cv2.GaussianBlur(current_image, (21,21), 0)
    current_image = dilate_erode(current_image, 0, 5)

    if previous_image_output is None:
        return False, current_image
    
    difference = cv2.absdiff(current_image, previous_image_output)
    _, thresh = cv2.threshold(difference,55,255,cv2.THRESH_BINARY)
    history.append(np.count_nonzero(thresh))
    history.pop(0)
    average = sum(history) / len(history)
    return average, current_image
   
    