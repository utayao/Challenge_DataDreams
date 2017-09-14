import cv2
import numpy as np
from  scipy.misc import imrotate


def fliplr(image,kwargs):
    return cv2.flip(image, 0)

def flipud(image,kwargs):
    return cv2.flip(image,1)

def rotate_image(image, kwargs):
    start, end = kwargs['start'],kwargs['end']
    deg = np.random.uniform(start,end,size=1)[0]
    return imrotate(image, deg)



def global_funcs(func):
    return globals()[func]
