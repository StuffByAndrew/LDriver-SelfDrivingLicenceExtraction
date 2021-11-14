import cv2
import numpy as np
from road_center_detection import get_roadcolor_center
from line_center_detection import get_roadline_center

def detect_lane_center(image):
    color_x, color_y = get_roadcolor_center(image)
    line_x, line_y = get_roadline_center(image)
    return ((color_x + line_x)//2, (color_y + line_y)//2)