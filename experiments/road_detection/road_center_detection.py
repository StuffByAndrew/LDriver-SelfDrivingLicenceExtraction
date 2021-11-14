import cv2 
import numpy as np
"""
for road detection:
uh = 0
us = 0
uv = 95
lh = 0
ls = 0
lv = 75
"""
def hsv_threshold(img, lh=0, uh=0, ls=0, us=0, lv=0, uv=0):
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

def mask(image, bottom_percent=1):
    top_percent = 1 - bottom_percent
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (0,int(top_percent*image.shape[0])), (image.shape[1], image.shape[0]), 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)

def threshold(image, lower_threshold, upper_threshold): 
    _, threshold = cv2.threshold(image, lower_threshold, upper_threshold, cv2.THRESH_BINARY)
    return threshold

def find_center(image):
    M = cv2.moments(image)
    try:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    except ZeroDivisionError as e:
        # NOTE: idk what to do here I just made it so the center is the middle of the image
        image_height, image_width, _ = image.shape
        return image_width//2, image_height//2
    return cx, cy

def get_roadcolor_center(image):
    center_image = mask(image, 0.3)
    center_image = hsv_threshold(center_image, lv=75, uv=95)
    center_image = threshold(center_image, lower_threshold=240, upper_threshold=255)
    return find_center(center_image)

if __name__ == "__main__":
    file = "experiments\\test_images\\distant.png"
    image = cv2.imread(file, cv2.IMREAD_COLOR)
    center = mask(image, 0.3)
    center = hsv_threshold(center, lv=75, uv=95)
    center = threshold(center, lower_threshold=240, upper_threshold=255)
    cv2.circle(image, find_center(center), 20, (255,0,0), -1)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


