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


class Generator(object):
    def __init__(self, label_path):
        self.label_path = label_path

    def read_images_to_arr(self, images, subset, labels=None):

        images_arr = []
        if subset:
            images = images[:10]
        for image_path in images:
            images_arr.append(utils.read_image(image_path))
        if labels:
            labels = [os.path.join(self.label_path, os.path.basename(i).replace(".tiff", ".png")) for i in images]
            if subset:
                labels = labels[:10]
            labels_arr = []
            for label_path in labels:
                labels_arr.append(utils.read_label(label_path, binary=True))
            return np.array(images_arr), np.array(labels_arr)
        return np.array(images_arr)


class TrainDataGenerator(Generator):
    def __init__(self, image_resize=None, subset=True, train_dir=None, cancer_data_augmentation=None,
                 non_cancer_data_augmentation=None, cv=2,
                 shuffle=True, normalize=False):
        assert train_dir is not None, "path should not be None"
        cancer_image_path = os.path.join(train_dir, "Image")
        label_path = os.path.join(train_dir, "result_labels")
        super(TrainDataGenerator, self).__init__(label_path=label_path)
        self._image_resize = image_resize
        self._cancer_data_augmentation = cancer_data_augmentation
        self._non_cancer_data_augmentation = non_cancer_data_augmentation
        self._shuffle = shuffle
        self._subset = subset
        self._normalize = normalize
        self._cv = cv
        non_cancer_images_path = glob.glob(os.path.join(cancer_image_path, "non_cancer_subset00/*.tiff"))
        cancer_images_path = glob.glob(os.path.join(cancer_image_path, "cancer_subset0[0-8]/*.tiff"))
        random.shuffle(cancer_images_path)
        random.shuffle(non_cancer_images_path)
        assert any([len(label_path) > 1, len(non_cancer_images_path) > 1,
                    len(cancer_images_path) > 1]), "list can not be empty"
        self.cancer_images, self.cancer_labels = self.read_images_to_arr(images=cancer_images_path, subset=subset,
                                                                         labels=True)
        self.non_cancer_images = self.read_images_to_arr(images=non_cancer_images_path, subset=subset, labels=False)
        self.cancer_train_indices, self.non_cancer_train_indices = None, None
        print self.cancer_images.shape, self.non_cancer_images.shape
        if cv:
            if not os.path.exists("../phase_1/non_cancer_splits.txt") or not os.path.exists(
                    "../phase_1/cancer_splits.txt"):
                self._splits = KFold(n_splits=cv, shuffle=True)
                self.cancer_train_indices = [(train_index, test_index) for train_index, test_index in
                                             self._splits.split(self.cancer_images)]
                self.non_cancer_train_indices = [(train_index, test_index) for train_index, test_index in
                                                 self._splits.split(self.non_cancer_images)]

                utils.save_obj(self.cancer_train_indices, "../phase_1/cancer_splits.txt")
                utils.save_obj(self.non_cancer_train_indices, "../phase_1/non_cancer_splits.txt")

            else:
                print "cross validation already available"
                self.cancer_train_indices = utils.load_obj("../phase_1/cancer_splits.txt")
                self.non_cancer_train_indices = utils.load_obj("../phase_1/non_cancer_splits.txt")
            assert self.cancer_train_indices or self.non_cancer_train_indices, " list can not be empty"
            print 'length of cancer indicies: {}, length of non cancer indicies: {}'.format(
                len(self.cancer_train_indices),
                len(
                    self.non_cancer_train_indices))
            print 'Number of cancer indices train in each cv {}'.format([len(i[0]) for i in self.cancer_train_indices])
            print 'Number of  cancer indices test in each cv {}'.format([len(i[1]) for i in self.cancer_train_indices])
            print 'Number of non cancer indices train in each cv {}'.format(
                [len(i[0]) for i in self.non_cancer_train_indices])
            print 'Number of non cancer indices in each cv {}'.format(
                [len(i[1]) for i in self.non_cancer_train_indices])

        self.cancer_image_counter = [0, 0]
        self.non_cancer_image_counter = [0, 0]

    def extract_patches(self, number_of_cancer_images=0, number_of_non_cancer_images=0, save_path=None):
        from tqdm import tqdm
        assert self._cv is None, "cv should be None"
        utils.makedir(save_path)
        counter = 0
        while counter < number_of_cancer_images:
            # utils.display(self.cancer_images[index,:])
            index = np.random.randint(low=0, high=self.cancer_images.shape[0], size=1)[0]
            cancer_image = self.cancer_images[index]
            cancer_label = self.cancer_labels[index]
            #pdb.set_trace()
            cancer_patches = utils.extract_random_patch_from_contour(image=cancer_image,
                                                                     label=cancer_label,
                                                                     patch_size=self._image_resize,
                                                                     max_patches=1,
                                                                     cancer_ratio=0.9)
            print cancer_patches.shape
            utils.save_array_of_images(cancer_patches, save_path, counter=index, label=1)
            counter += 1
            del cancer_patches
        counter = 0
        while counter < number_of_non_cancer_images:
            non_cancer_patches = sklearn_image.extract_patches_2d(self.non_cancer_images[index, :],
                                                                  patch_size=self._image_resize,
                                                                  max_patches=1)
            utils.save_array_of_images(non_cancer_patches, save_path, counter=index, label=0)
            print  counter
            counter += 1
        del non_cancer_patches

    def sample_batch(self, batch_size, cv, cancer_ratio=0.5, index=0):
        if index == 1:
            print "index is 1"
        cancer_batch_size = int(cancer_ratio * batch_size)
        non_cancer_batch_size = batch_size - cancer_batch_size
        cv_cancer_images, cv_cancer_labels = self.cancer_images[self.cancer_train_indices[cv][index]], \
                                             self.cancer_labels[
                                                 self.cancer_train_indices[cv][index]]
        cv_non_cancer_images = self.non_cancer_images[self.non_cancer_train_indices[cv][index]]

        cv_cancer_images, cv_cancer_labels = sklearn.utils.shuffle(cv_cancer_images, cv_cancer_labels)
        cv_non_cancer_images = sklearn.utils.shuffle(cv_non_cancer_images)
        # pdb.set_trace()
        print "cancer counter ", self.cancer_image_counter

        print "non cancer counter ", self.non_cancer_image_counter

        cancer_images, cancer_labels, self.cancer_image_counter[index] = utils.extract_patches(images=cv_cancer_images,
                                                                                               labels=cv_cancer_labels,
                                                                                               max_patch=cancer_batch_size,
                                                                                               patch_size=self._image_resize,
                                                                                               counter=
                                                                                               self.cancer_image_counter[
                                                                                                   index])
        non_cancer_images, non_cancer_labels, self.non_cancer_image_counter[index] = utils.extract_patches(
            images=cv_non_cancer_images, labels=None,
            max_patch=non_cancer_batch_size, patch_size=self._image_resize,
            counter=self.non_cancer_image_counter[index])
        self.cancer_image_counter[index] %= len(self.cancer_train_indices[cv][index])
        self.non_cancer_image_counter[index] %= len(self.non_cancer_train_indices[cv][index])

        if self._cancer_data_augmentation:
            for index in range(len(cancer_images)):
                img = cancer_images[index]
                if np.random.uniform() > 0.35:
                    for func, params in self._cancer_data_augmentation.items():
                        if np.random.uniform() > 0.5:
                            img = global_funcs(func)(img, params)
                cancer_images[index] = img if type(img).__module__ == np.__name__ else tensor_eval(img)
        img = None
        if self._non_cancer_data_augmentation:
            for index in range(len(non_cancer_images)):
                img = non_cancer_images[index]
                if np.random.uniform() > 0.35:
                    for func, params in self._non_cancer_data_augmentation.items():
                        if np.random.uniform() > 0.5:
                            img = global_funcs(func)(img, params)
                non_cancer_images[index] = img if type(img).__module__ == np.__name__ else tensor_eval(img)

        images = np.concatenate((cancer_images, non_cancer_images), axis=0)
        labels = np.concatenate((cancer_labels, non_cancer_labels), axis=0)
        if self._shuffle:
            images, labels = sklearn.utils.shuffle(images, labels)
        if self._normalize:
            for i, img in enumerate(images):
                images[i] = utils.normalize(img, between=[-1, 1])
        # labels = utils.one_hot_vector(labels)
        # print labels

        return images, utils.one_hot_vector(labels)
