import os
import cv2
import numpy as np
from sklearn.feature_extraction import image as sklearn_image
import pickle
from scipy import misc
from scipy import ndimage


def flood_fill(label):
    th, im_th = cv2.threshold(label, 0, 127, cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv
    return cv2.bitwise_not(im_out)


def save_image(img, save_path):
    misc.imsave(save_path, img)


def read_image(img_path, size=None):
    img = cv2.imread(img_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size:
        assert isinstance(size, tuple), "Size is not in tuple"
        return misc.imresize(img, size)
    return img


def read_label(mask_path, binary=False, size=None):
    img = ndimage.imread(mask_path, mode='L')
    if size:
        assert isinstance(size, tuple), "Size is not in tuple"
        img = misc.imresize(img, size)
    if binary:
        img[img <= 127] = 1
        img[img > 127] = 0
        return img
    return img


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def intersection(img, contour):
    print img, contour


def save_obj(obj, obj_path):
    makedir(os.path.dirname(obj_path))
    with open(obj_path, "wb") as f:
        pickle.dump(obj, f)


def load_obj(obj_path):
    with open(obj_path, "rb") as f:
        return pickle.load(f)


def extract_patches_2d(image, bounding_box, label, patch_size=(224, 224), max_patches=1):
    x, y, w, h = bounding_box

    rand_x, rand_y, w, h = np.random.randint(low=x, high=x + w, size=1)[0], \
                           np.random.randint(low=y, high=y + h, size=1)[0], patch_size[0], patch_size[1]
    label_image = label[rand_y:rand_y + h, rand_x:rand_x + w]
    intersection = np.count_nonzero(label_image) * 1.0 / (
    np.count_nonzero(label_image) + len(np.where(label_image == 0)))
    # return label_image,intersection
    print rand_x,rand_y,w,h
    return image[rand_y:rand_y + h, rand_x:rand_x + w, :], intersection


def extract_random_patch_from_contour(image, label, patch_size, max_patches, cancer_ratio):
    _, contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    images = []
    counter = 0
    while counter < max_patches:
        index = np.random.randint(0, len(bounding_boxes))
        bounding_box_image = bounding_boxes[index]
        x, y, w, h = bounding_box_image
        image_bounding_box = image[y:y + h, x:x + w, :]
        img, intersection = extract_patches_2d(image, bounding_box_image, label, patch_size=patch_size,
                                               max_patches=1)
        if intersection > cancer_ratio:
            print img.shape
            images.append(img)
            counter += 1

    return np.array(images)


def extract_patches(images, labels=None, max_patch=1, patch_size=(224, 224), counter=0, **kwargs):
    if labels is not None:
        cancer_ratio = kwargs.get("cancer_ratio", 0.75)
        images = extract_random_patch_from_contour(images[counter], labels[counter], patch_size=patch_size,
                                                   max_patches=max_patch,
                                                   cancer_ratio=cancer_ratio)
        labels = np.ones_like(np.arange(max_patch, dtype=np.float))
    else:
        images = sklearn_image.extract_patches_2d(images[counter], patch_size=patch_size,
                                                  max_patches=max_patch)
        labels = np.zeros_like(np.arange(max_patch, dtype=np.float))

    return images, labels, (counter + max_patch) % len(images)
