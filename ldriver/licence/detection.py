import cv2
import imutils
import numpy as np
from operator import itemgetter
from hsv_config import licence_ranges
from functools import reduce

def hsv_threshold(img):
    """ Thresholds img by hsv range defined in hsv_config

    Args:
        img (numpy.ndarray): image matrix

    Returns:
        numpy.ndarray: thresholded image
    """
    out_img = img.copy()
    hsv = cv2.cvtColor(out_img, cv2.COLOR_BGR2HSV)
    cv2.medianBlur(hsv,5)
    msks = [cv2.inRange(hsv, l, u) for l, u in licence_ranges.get_ranges()]
    out_img = reduce((lambda x, y: cv2.bitwise_or(x,y)), msks)
    return out_img

def rect_contours(img, orig_img=None, min_area=800):
    """ Finds large rectangular contours. 

    Args:
        img (numpy.ndarray): processed version of orig_img that will be used to detect contours
        orig_img (numpy.ndarray): original, unaltered version that found contours will be drawn on
        min_area (int): all rectangles that have a smaller area will be filtered out

    Returns:
        numpy.ndarray: Contours detected (Opencv2 style)
        numpy.ndarray: orig_img with contours drawn on (Red)
    """
    out_img = orig_img.copy() if orig_img is not None else img.copy()
    
    # Grab largest contours in reverse area order
    edged = cv2.Canny(img.copy(), 30, 200) 
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse = True)[:5] # 5 largest

    # Approximate rectangular contours
    def approx_rect(c):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        return approx

    # filter non-rectangular, and too small contours
    best_conts = np.array(list(
        filter(lambda c: cv2.contourArea(c) > min_area,
        filter(lambda x: len(x)==4, 
        map(approx_rect, 
        contours)))))
    cv2.drawContours(out_img, best_conts, -1, (0, 0, 255), 3)

    return best_conts, out_img

def combine_rects(contours, orig_img=None):
    """Combine the two rectangles if there is need

    Args:
        contours (numpy.ndarray): Opencv2 style contours
        orig_img (numpy.ndarray, optional): original image to draw final rectangle on. Defaults to None.

    Returns:
        numpy.ndarray: points denoting a rectangle
        numpy.ndarray: orig_img with rectangle corners drawn on (Green)
    """
    out_img = orig_img.copy() if orig_img is not None  else None
    # 
    rect_pts = np.array([])
    if contours.shape[0] > 2:
        all_pts = contours.reshape(-1, contours.shape[-1]) if contours.size else contours
        all_pts = sorted(all_pts, key=itemgetter(1))
        rect_pts = np.int32(all_pts[:4:2]+all_pts[-4::2])
        line_pts = rect_pts.reshape((-1, 1, 2))
        cv2.polylines(orig_img, line_pts, True, (0, 255 ,0), 3)

    return rect_pts, out_img

def sort_rect_pts(pts):
    """ Given points that are the corners of a rectangle sort by left to right first, then top to bottom

    Args:
        pts (iterable): array of (x,y) points

    Returns:
        numpy.float32: list of ordered points
    """

    # This only works if the right two points and the left two points never cross axes
    # sort by x
    x_sort = sorted(pts, key=itemgetter(0))
    left_pts, right_pts = x_sort[:2], x_sort[-2:]
    
    # sort by y
    left_pts, right_pts = sorted(left_pts, key=itemgetter(1)), sorted(right_pts, key=itemgetter(1))

    return np.float32(left_pts+right_pts)
    
def warp_rect(img, in_pts, out_pts=np.float32([[0,0],[300,0],[0,300],[300,300]]), size=(300,300)):
    """ Using rectangular corner coordinates, warps the shape specified by in_pts in img to a new image
        of size size specified by out_pts

    Args:
        img (numpy.ndarray): original image
        in_pts (numpy.float32): 4 points on img to be cropped and warped
        out_pts (numpy.float32, optional): 4 points on returned image to be warped to. 
        Defaults to np.float32([[0,0],[300,0],[0,300],[300,300]]).
        size (tuple, optional): size of output image. Defaults to (300,300).

    Returns:
        numpy.ndarray: output image.
    """
    # To match all points by sorting cartesianly
    inpts, outpts = sort_rect_pts(in_pts), sort_rect_pts(out_pts)
    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(inpts, outpts)

    # Apply the perspective transformation to the image
    dest = cv2.warpPerspective(img,M,size)
    return dest

def dilate_erode(img):
    """ Denoise salt and pepper noise in an image through dilation and erosion

    Args:
        img (numpy.ndarray): binary image matrix

    Returns:
        numpy.ndarray: denoised binary image
    """
    kernel5 = np.ones((5,5), np.uint8)
    kernel3 = np.ones((3,3), np.uint8)
    img = cv2.dilate(img, kernel5, iterations=1)
    img = cv2.erode(img, kernel5, iterations=1)
    img = cv2.erode(img, kernel3, iterations=2)
    img = cv2.dilate(img, kernel5, iterations=2)
    return img

def find_licence(image):
    """Find licence from image

    Args:
        image (numpy.ndarray): input image matrix

    Returns:
        numpy.ndarray: extracted licence image if found, empty array if not.
    """
    orig_img = image
    thresh = hsv_threshold(orig_img)
    thresh = dilate_erode(thresh)
    conts, img = rect_contours(thresh, orig_img)
    if conts.size:
        pts, img = combine_rects(conts, img)
        new_img = warp_rect(orig_img, pts)
        return new_img
    return np.array([])

if __name__ == '__main__':
    # Testing
    import glob
    # orig_img = cv2.imread('./experiments/test_images/74.png')
    for img in glob.glob('experiments/test_images/*.png'):
        orig_img = cv2.imread(img)
        cv2.imshow('testing', orig_img)
        cv2.waitKey(0)

        thresh = hsv_threshold(orig_img)
        thresh = dilate_erode(thresh)
        cv2.imshow('testing', thresh)
        cv2.waitKey(0)

        conts, img = rect_contours(thresh, orig_img)
        cv2.imshow('testing', img)
        cv2.waitKey(0)

        pts, img = combine_rects(conts, img)
        cv2.imshow('testing', img)
        cv2.waitKey(0)

        try:
            new_img = warp_rect(orig_img, pts)
            cv2.imshow('testing', new_img)
            cv2.waitKey(0)
        except:
            print('no licence')

    from utils import BoundingBoxWidget
    bbwidget = BoundingBoxWidget(new_img)
    bbwidget.display()