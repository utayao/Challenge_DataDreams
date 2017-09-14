import cv2
import numpy as np
import tensorflow as tf
from scipy import ndimage


def rotate_image(img, kwargs):
    low_angle, high_angle = kwargs.get("angle", [1, 5])
    angle = np.random.uniform(low=low_angle, high=high_angle, size=(1,))[0]
    return tf.image.rot90(img, angle)


def flipud(img, kwargs):
    return tf.image.flip_left_right(img)


def fliplr(img, kwargs):
    return tf.image.flip_up_down(img)


def adjust_brightness(img, kwargs):
    return tf.image.adjust_brightness(img, delta=(64.0 / 255))


def adjust_contrast(img, kwargs):
    return tf.image.adjust_contrast(img, 0.75)


def adjust_hue(img, kwargs):
    return tf.image.adjust_hue(img, 0.04)


def adjust_saturation(img, kwargs):
    return tf.image.adjust_saturation(img, 0.25)

def normalize(img,between=[-1,1]):
    a,b = between
    return ((b-a)* (img - np.min(img))/(np.max(img) - np.min(img))) + a

def tensor_eval(img):
    with tf.Session() as sess:
        with sess.as_default():
            img = img.eval()
    return img

def global_funcs(func):
    return globals()[func]
