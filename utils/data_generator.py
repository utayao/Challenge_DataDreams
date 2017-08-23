import os
import glob
import numpy as np
import sklearn
import utils


class Generator(object):
    @staticmethod
    def read_images_to_arr(images, subset, labels=None):
        images_arr = []
        if subset:
            images = images[:100]
        for image_path in images:
            images_arr.append(utils.read_image(image_path))
        if labels:
            if subset:
                labels = labels[:100]
            labels_arr = []
            for label_path in labels:
                labels_arr.append(utils.read_label(label_path, binary=True))
            return images_arr, labels_arr
        return images_arr


class TrainDataGenerator(Generator):
    def __init__(self, image_resize=(224, 224), subset=True, train_dir=None, cancer_data_augmentation=None,
                 non_cancer_data_augmentation=None, cv=2,
                 shuffle=True, normalize=False):
        assert train_dir is not None, "path should not be None"
        cancer_image_path = os.path.join(train_dir, "Image")
        label_path = os.path.join(train_dir, "results_labels")
        self._image_resize = image_resize
        self._cancer_data_augmentation = cancer_data_augmentation
        self._non_cancer_data_augmentation = non_cancer_data_augmentation
        self._shuffle = shuffle
        self._subset = subset
        cancer_image_path = np.sort(glob.glob(os.path.join(cancer_image_path, "cancer_subset0[0-8]/*.tiff")))
        non_cancer_image_path = np.sort(glob.glob(os.path.join(cancer_image_path, "non_cancer_subset/*.tiff")))
        label_path = np.sort(glob.glob(os.path.join(label_path, "*.png")))
        assert label_path or non_cancer_image_path or cancer_image_path, "list can not be empty"
        self.cancer_images, self.cancer_labels = self.read_images_to_arr(images=cancer_image_path, subset=subset,
                                                                         labels=label_path)
        self.non_cancer_images = self.read_images_to_arr(images=non_cancer_image_path, subset=subset)
        self.cancer_image_counter = 0
        self.non_cancer_image_counter = 0

    def sample_batch(self, batch_size, cancer_ratio=0.5):
        cancer_batch_size = int(cancer_ratio * batch_size)
        non_cancer_batch_size = batch_size - cancer_batch_size
        cancer_images, cancer_labels, self.cancer_image_counter = utils.extract_patches(images=self.cancer_images,
                                                                                        labels=self.cancer_labels,
                                                                                        max_patch=cancer_batch_size,
                                                                                        patch_size=self._image_resize,
                                                                                        counter=self.cancer_image_counter)
        non_cancer_images, non_cancer_labels, self.non_cancer_image_counter = utils.extract_patches(
            images=self.non_cancer_images, label=None,
            max_patch=non_cancer_batch_size, patch_size=self._image_resize,
            counter=self.non_cancer_image_counter)
        if self._cancer_data_augmentation:
            for index in range(len(cancer_images)):
                for func, params in self._cancer_data_augmentation.items():
                    cancer_images[index] = func(cancer_images[index], params)
        if self._non_cancer_data_augmentation:
            for index in range(len(non_cancer_images)):
                for func, params in self._non_cancer_data_augmentation.items():
                    non_cancer_images[index] = func(non_cancer_images[index], params)

        images = np.hstack((cancer_images, non_cancer_images))
        labels = np.hstack((cancer_labels, non_cancer_labels))
        if self._shuffle:
            images, labels = sklearn.utils.shuffle(images, labels)
        return images, labels
