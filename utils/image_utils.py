import cv2
import numpy as np

def crop_by_bbox(image, bbox, image_size):
    """
    BBOX: [ymin, xmin, ymax, xmax] normalized format (0-1)
    """
    h, w = image_size
    y1 = int(bbox[0] * h)
    x1 = int(bbox[1] * w)
    y2 = int(bbox[2] * h)
    x2 = int(bbox[3] * w)
    return image[y1:y2, x1:x2]
