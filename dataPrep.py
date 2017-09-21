import cv2
import pdb
import os
import sys
import PIL
import h5py
import sys
import glob
import numpy as np
import time
# import nn.image_processing as ip
from tqdm import tqdm
import initialization
from sklearn.utils import shuffle
from PIL import Image
import tensorflow as tf
import utils
from data_gen3 import TrainGenrator
from data_generator import TrainDataGenerator
import logging

flags = tf.app.flags
FLAGS = flags.FLAGS
CLASSES = initialization.classes
flags.DEFINE_integer("size", 224, "reduced dimension")
flags.DEFINE_bool("subset", True, "subset")
flags.DEFINE_string("train_path", "/home/ashwin/Challenge_DataDreams/data", "train file")
flags.DEFINE_integer("scaled_size", 224, "scaled height")


def display(epoch, **kwargs):
    backup = sys.stdout
    sys.stdout = LogToFile()
    print "#" * 50
    print "===> Epoch: {}".format(epoch)
    for name, value in kwargs.items():
        print "{}={}".format(name, value)
    print "#" * 50
    sys.stdout = backup


class LogToFile(object):
    def write(self, s):
        sys.__stdout__.write(s)
        open("%s.log" % (FLAGS.model_name), "a").write(s)


def load_data(path):
    #    pdb.set_trace()
    images_path = glob.glob(os.path.join(path, 'patches/*.jpg'))
    images_arr = []
    labels_arr = []
    if FLAGS.subset:
        images_path = images_path[:50]
    for image_path in images_path:
        label = int(os.path.basename(image_path).replace('.jpg', '').split('_')[-1])
        images_arr.append(utils.read_image(image_path))
        labels_arr.append(label)
    return np.array(images_arr), utils.one_hot_vector(labels_arr)


if __name__ == "__main__":
    td = TrainGenrator('/home/ashwin/Challenge_DataDreams/data/patches_new', subset=False)
    img, label = td.next_batch(64,True)
    img, label = td.next_batch(64,True)
    img, label = td.next_batch(64,True)

    print td.train_queue.empty()
