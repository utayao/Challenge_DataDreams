import os
import pdb
import glob
import numpy as np
import sklearn
from sklearn.model_selection import KFold
import utils


class Generator(object):

    def __init__(self,label_path):
        self.label_path = label_path

    def read_images_to_arr(self,images, subset, labels=None):

        images_arr = []
        if subset:
            images = images[:10]
        for image_path in images:
            images_arr.append(utils.read_image(image_path))
        if labels:
            labels = [os.path.join(self.label_path,os.path.basename(i).replace(".tiff",".png")) for i in images]
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
        non_cancer_images_path = glob.glob(os.path.join(cancer_image_path, "non_cancer_subset00/*.tiff"))
        cancer_images_path = glob.glob(os.path.join(cancer_image_path, "cancer_subset0[0-8]/*.tiff"))
        label_path = glob.glob(os.path.join(label_path, "*.png"))
        assert any([len(label_path) >1,len(non_cancer_images_path) >1, len(cancer_images_path)>1]), "list can not be empty"
        self.cancer_images, self.cancer_labels = self.read_images_to_arr(images=cancer_images_path, subset=subset,
                                                                         labels=True)
        self.non_cancer_images = self.read_images_to_arr(images=non_cancer_images_path, subset=subset, labels=False)
        self.cancer_train_indices, self.non_cancer_train_indices = None, None
        print self.cancer_images.shape,self.non_cancer_images.shape
        if not os.path.exists("../phase_1/non_cancer_splits.txt") or not os.path.exists("../phase_1/cancer_splits.txt"):
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
        self.cancer_image_counter = 0
        self.non_cancer_image_counter = 0
        
    def sample_batch(self, batch_size, cv, cancer_ratio=0.5, index=0):

        cancer_batch_size = int(cancer_ratio * batch_size)
        non_cancer_batch_size = batch_size - cancer_batch_size
        cv_cancer_images, cv_cancer_labels = self.cancer_images[self.cancer_train_indices[cv][index]], \
                                             self.cancer_labels[
                                                 self.cancer_train_indices[cv][index]]
        cv_non_cancer_images = self.non_cancer_images[self.non_cancer_train_indices[cv][index]]

        #pdb.set_trace()
        self.cancer_image_index = self.cancer_train_indices[cv][index][self.cancer_image_counter]
        self.non_cancer_image_index = self.non_cancer_train_indices[cv][index][self.non_cancer_image_counter]
        #print self.cancer_image_counter
        #print self.non_cancer_image_counter
        #print self.cancer_image_index
        #print self.non_cancer_image_index

        cancer_images, cancer_labels, self.cancer_image_counter = utils.extract_patches(images=cv_cancer_images,
                                                                                        labels=cv_cancer_labels,
                                                                                        max_patch=cancer_batch_size,
                                                                                        patch_size=self._image_resize,
                                                                                        index=self.cancer_image_index,
                                                                                       counter=self.cancer_image_counter)
        non_cancer_images, non_cancer_labels, self.non_cancer_image_counter = utils.extract_patches(
            images=cv_non_cancer_images, labels=None,
            max_patch=non_cancer_batch_size, patch_size=self._image_resize,
            index=self.non_cancer_image_index,
            counter=self.non_cancer_image_counter)
                
        if self._cancer_data_augmentation:
            for index in range(len(cancer_images)):
                for func, params in self._cancer_data_augmentation.items():
                    cancer_images[index] = func(cancer_images[index], params)
        if self._non_cancer_data_augmentation:
            for index in range(len(non_cancer_images)):
                for func, params in self._non_cancer_data_augmentation.items():
                    non_cancer_images[index] = func(non_cancer_images[index], params)

        images = np.concatenate((cancer_images, non_cancer_images),axis=0)
        labels = np.concatenate((cancer_labels, non_cancer_labels),axis=0)
        if self._shuffle:
            images, labels = sklearn.utils.shuffle(images, labels)
        if self._normalize:
            for i, img in enumerate(images):
                images[i] = utils.normalize(img, between=[-1, 1])
        #labels = utils.one_hot_vector(labels)
        #print labels
        return images, utils.one_hot_vector(labels)
