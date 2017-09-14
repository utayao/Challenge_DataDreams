import os
import pdb
import glob
import numpy as np
import sklearn
import random
from sklearn.feature_extraction import image as sklearn_image
from sklearn.model_selection import KFold
from data_augmentation import global_funcs, tensor_eval

import utils


class TrainDataGenerator(object):
    def __init__(self, image_resize=None, subset=True, train_dir=None, data_augmentation=None, cv=2, shuffle=True,
                 normalize=False):
        images_path = glob.glob(os.path.join(train_dir, 'patches/*.jpg'))
        self._image_resize = image_resize
        self._data_augmentation = data_augmentation
        self._shuffle = shuffle
        self._subset = subset
        self._normalize = normalize
        self._cv = cv
        random.shuffle(images_path)
        self.images, self.labels = self.read_images(images_path, subset=subset)
        if cv:
            if not os.path.exists('../phase_1/splits_subset_{}.txt'.format(subset)):
                self._splits = KFold(n_splits=cv, shuffle=True)
                self.indicies = [(train_index, test_index) for train_index, test_index in
                                 self._splits.split(self.images)]
                utils.save_obj(self.indicies, '../phase_1/splits_subset_{}.txt'.format(subset))
            else:
                print 'cross validation avaliable'
                self.indicies = utils.load_obj('../phase_1/splits_subset_{}.txt'.format(subset))

            print 'Number of indices train in each cv {}'.format([len(i[0]) for i in self.indicies])
            print 'Number of indices test in each cv {}'.format([len(i[1]) for i in self.indicies])

        self.counter = 0

    def read_images(self, images, subset):

        images_arr = []
        labels_arr = []
        if subset:
            images = images[:10]
        for image_path in images:
            label = int(os.path.basename(image_path).replace('.jpg', '').split('_')[-1])
            images_arr.append(utils.read_image(image_path))
            labels_arr.append(label)

        return np.array(images_arr), np.array(labels_arr)

    def sample_batch(self, batch_size, each_cv, index):

        cv_images, cv_labels = self.images[self.indicies[each_cv][index]], self.labels[self.indicies[each_cv][index]]

        cv_images_batch, cv_labels_batch = cv_images[self.counter: min(cv_images.shape[0],self.counter + batch_size)], cv_labels[self.counter: min(cv_images.shape[0], self.counter + batch_size)]
        self.counter = 0 if self.counter > cv_images.shape[0] else self.counter + batch_size

        if self._data_augmentation:
            for index in range(cv_images_batch.shape[0]):
                img = cv_images_batch[index]
                if np.random.uniform() > 0.35:
                    for func, params in self._data_augmentation.items():
                        if np.random.uniform() > 0.5:
                            img = global_funcs(func)(img, params)
                cv_images_batch[index] = img if type(img).__module__ == np.__name__ else tensor_eval(img)
            if self._shuffle:
                cv_images_batch, cv_labels_batch = sklearn.utils.shuffle(cv_images_batch, cv_labels_batch)
        if self._normalize:
            for i, img in enumerate(cv_images_batch):
                cv_images_batch[i] = utils.normalize(img, between=[-1, 1])
        # labels = utils.one_hot_vector(labels)
        # print labels

        return cv_images_batch, utils.one_hot_vector(cv_labels_batch)
