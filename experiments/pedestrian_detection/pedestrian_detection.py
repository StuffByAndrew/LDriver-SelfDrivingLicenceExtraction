import sys
import cv2
import numpy as np
import imutils
from ldriver.steering.lane_detection import hsv_threshold
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

def pedestrian_detected(input_image):
    mask = get_road_mask(input_image) 
    image = cv2.bitwise_and(input_image, input_image, mask=mask)
    image = hsv_threshold(image, lh=0, ls=10, lv=90, uh=0, us=255, uv=255)
    image = dilate_erode(image, 0, 5)
    image = dilate_erode(image, 5, 0)
    edged = cv2.Canny(image.copy(), 30, 200) 
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return True if len(contours) > 4 else False

if __name__ == "__main__":
    import glob
    for image_file in glob.glob('img_cmd_data/*.png'):
        image = cv2.imread(image_file)
        print(pedestrian_detected(image))
        print(image_file)
        cv2.imshow("image", image)
        cv2.waitKey(0)
    