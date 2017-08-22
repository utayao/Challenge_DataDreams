import os
import glob
import numpy as np
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
    def __init__(self, image_resize=(224, 224), subset=True, train_dir=None, data_augmentation=None, cv=2,
                 shuffle=True, normalize=False):
        assert train_dir is not None, "path should not be None"
        cancer_image_path = os.path.join(train_dir, "Image")
        label_path = os.path.join(train_dir, "results_labels")
        self._image_resize = image_resize
        self._data_augmentation = data_augmentation
        self._shuffle = shuffle
        self._subset = subset
        cancer_image_path = np.sort(glob.glob(os.path.join(cancer_image_path, "cancer_subset0[0-8]/*.tiff")))
        non_cancer_image_path = np.sort(glob.glob(os.path.join(cancer_image_path, "non_cancer_subset/*.tiff")))
        label_path = np.sort(glob.glob(os.path.join(label_path, "*.png")))
        assert label_path or non_cancer_image_path or cancer_image_path, "list can not be empty"
        self.cancer_images, self.cancer_labels = self.read_images_to_arr(images=cancer_image_path, subset=subset,
                                                                         labels=label_path)
        self.non_cancer_images = self.read_images_to_arr(images=non_cancer_image_path, subset=subset)
        print self.cancer_images.shape

    def sample_batch(self, batch_size, cancer_ratio=0.5):
        noncancer_ratio = 1 - cancer_ratio
