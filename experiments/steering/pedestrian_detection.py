import sys
import cv2
import time
import numpy as np
import imutils
from ldriver.steering.lane_detection import hsv_threshold, mask_rectangle, find_center
# from dnldriver.licence.detection import dilate_erode
# from ldriver.steering.lane_detection import hsv_threshold

"""
Red line detection:
Lower HSV: [0, 10, 90]
Upper HSV: [0, 255, 255]
"""
"""
road detection:
Lower HSV: [0, 0, 75]
Upper HSV: [0, 0, 95]
"""

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

def pedestrian_crossing(current_image_input, previous_image_output):
    current_image = cv2.cvtColor(current_image_input, cv2.COLOR_BGR2GRAY)
    current_image = cv2.GaussianBlur(current_image, (25,25), 0)

    if previous_image_output is None:
        return False, current_image
    
    difference = cv2.absdiff(current_image, previous_image_output)
    _, thresh = cv2.threshold(difference,55,255,cv2.THRESH_BINARY)
    cv2.imshow("image", difference)
    cv2.waitKey(0)
    return np.any(thresh), current_image
   

def redline_detected(input_image, bottom_percent=1):
    """returns true if red line is detected within the bottom_percent of the image"""
    start = time.time()
    #mask = get_road_mask(input_image)
    #image = cv2.bitwise_and(input_image, input_image, mask=mask)
    image = mask_rectangle(input_image, left=0.2, top=(1-bottom_percent), right=0.2, bottom=0)
    image = hsv_threshold(image, lh=0, ls=10, lv=90, uh=0, us=255, uv=255)
    print(time.time() - start)
    return np.any(image)

if __name__ == "__main__":
    import glob
    #for image_file in glob.glob("src/data_collection/scripts/img_cmd_data/*.png"):
    prev_image = None
    for image_file in glob.glob('experiments/pedestrian_detection/crossing/*.png'):
        image = cv2.imread(image_file)
        shouldpass, prev_image = pedestrian_crossing(image, prev_image)
        
        
        #image = cv2.imread(image_file)
        #image = cv2.imread("src/data_collection/scripts/img_cmd_data/124.png")
        #print(redline_detected(image, 0.26))
    