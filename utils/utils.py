import os
import cv2
import pdb
import uuid
import time
import numpy as np
from sklearn.feature_extraction import image as sklearn_image
import pickle
import matplotlib as mpl
from functools import partial
import json

mpl.use('Agg')
import matplotlib.pyplot as plt

plt.ioff()
from scipy import misc
from scipy import ndimage
import traceback

logging_instance = {}


def logger_func(path):
    global logging_instance
    instance = logging_instance.get(path)
    if not instance:
        import logging
        logger = logging.getLogger('log_file')
        hdlr = logging.FileHandler(path)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)
        logging_instance[path] = logger
        return logger
    else:
        return instance


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


def save_obj(obj, obj_path):
    makedir(os.path.dirname(obj_path))
    with open(obj_path, "wb") as f:
        pickle.dump(obj, f)


def display(img):
    misc.imshow(img)
    plt.show()


def load_obj(obj_path):
    with open(obj_path, "rb") as f:
        return pickle.load(f)


def log_to_file(path, txt):
    logger = logger_func(path)
    logger.info(txt)


def get_json(path):
    assert os.path.exists(path), "No such file exist"
    with open(path) as data_file:
        data = json.loads(data_file.read())
    return data


def get_state(path, key):
    data = get_json(path)
    return data.get(key, None)


def save_json(path, key, value, Type="scalar"):
    data = get_json(path)
    if Type.lower() == "list":
        if key not in data:
            data[key] = []
        data[key].append(str(value))
    else:
        data[key] = str(value)
    with open(path, "w") as data_file:
        json.dump(data, data_file)


def save_image(img_arr, path):
    misc.imsave(path, img_arr)


def one_hot_vector(labels):
    labels = np.array(labels, dtype=np.int)
    res = np.zeros((labels.size, int(labels.max()) + 1), dtype=np.int64)
    res[np.arange(labels.size), labels] = 1
    return res


def save_array_of_images(images_arr, data_path, counter, label):
    for i in range(images_arr.shape[0]):
        save_image(images_arr[i, :], os.path.join(data_path, str(counter) + '_' + str(i) + '_' + str(label) + '.jpg'))


def extract_patches_2d_new(image, label, bounding_box, patch_size=(224, 224), max_patches=1, cancer_ratio=0.9):
    x, y, w, h = bounding_box
    img = image[y: y + h, x:x + w]
    label_image = label[y: y + h, x:x + w]
    try:

        images = sklearn_image.extract_patches_2d(label_image, patch_size=patch_size,
                                                  max_patches=max_patches)
        # print images.shape
        #intersection = np.count_nonzero(images[0]) * 1.0 / (patch_size[0] * patch_size[1])
    except:
        return None

    return images


def extract_patches_2d(image, label, bounding_box, patch_size=(224, 224), max_patches=1):
    x, y, w, h = bounding_box
    rand_x, rand_y, w, h = np.random.randint(low=x, high=x + w, size=1)[0], \
                           np.random.randint(low=y, high=y + h, size=1)[0], patch_size[0], patch_size[1]
    if (rand_y + h > image.shape[1]):
        rand_y -= (rand_y + h - image.shape[1])
    if (rand_x + w > image.shape[0]):
        rand_x -= (rand_x + w - image.shape[0])

    label_image = label[rand_y:rand_y + h, rand_x:rand_x + w]
    intersection = np.count_nonzero(label_image) * 1.0 / (patch_size[0] * patch_size[1])
    return image[rand_y:rand_y + h, rand_x:rand_x + w, :], intersection, rand_x, rand_y


def extract_random_patch_from_contour(image, label, patch_size, max_patches, cancer_ratio):
    _, contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    images = []
    counter = 0
    # while counter < max_patches:
    start = time.time()
    while True:
        index = np.random.randint(0, len(bounding_boxes))
        bounding_box_image = bounding_boxes[index]
        img, intersection, rand_x,rand_y = extract_patches_2d(image, label, bounding_box_image, patch_size=patch_size,
                                                       max_patches=1)

        if intersection > cancer_ratio:
            print intersection
            label_image = label[rand_y:rand_y+patch_size[1],rand_x:rand_x+patch_size[0]]
            save_image(label_image * 255, str(uuid.uuid4()) + '.png')
            images.append(img)
            break
        if time.time() - start > 30:
            print 'time out'
            break

    return np.array(images)


def extract_patches(images, labels=None, max_patch=1, patch_size=(224, 224), counter=0, **kwargs):
    if labels is not None:
        cancer_ratio = kwargs.get("cancer_ratio", 0.90)
        images = extract_random_patch_from_contour(images[counter], labels[counter], patch_size=patch_size,
                                                   max_patches=max_patch,
                                                   cancer_ratio=cancer_ratio)
        labels = np.ones_like(np.arange(max_patch, dtype=np.float))
    else:
        images = sklearn_image.extract_patches_2d(images[counter], patch_size=patch_size,
                                                  max_patches=max_patch)
        labels = np.zeros_like(np.arange(max_patch, dtype=np.float))
    return images, labels, (counter + 1)
