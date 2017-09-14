import os
import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf
from model_camelyon import Model
from utils.data_augmentation import normalize


class Evaluator(object):
    def __init__(self, model_path,n_classes=2, test_data_path=None, patch_size=(224, 224)):
        self._net = Model(patch_size, n_classes)
        self._model_path = model_path
        self.patch_size = patch_size
        # self._test_data = TestDataLoader(test_data_path)
        self._saver = tf.train.import_meta_graph(model_path + '.meta')
        self.global_step = None
        self.build()
    def build(self, sess=None):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = sess or tf.Session(config=config)

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self._model_path))
        assert ckpt and ckpt.model_checkpoint_path, 'No checkpoint or model exists'
        self._saver.restore(self.sess, ckpt.model_checkpoint_path)
        self.global_step = os.path.basename(ckpt.model_checkpoint_path).split('-')[-1]
        self._net.build(train=False)
        self.sess.run(tf.global_variables_initializer())
        self.evaluation_ops = (list(self._net.validation_infer_ops))

    # def batch_data(self):
    #     im_data = self._test_data.read_data()
    #     return im_data, None

    def feed_dict(self, data, train):
        data_op, _, phase_train = self._net.input_ops
        return {data_op: data,phase_train: train}

    def build_heatmap(self, image, stride=0):
        heat_map = np.zeros((image.shape[0], image.shape[1], 2), dtype=np.float)
        for x in range(0, image.shape[0], stride):
            for y in range(0, image.shape[1], stride):
                if x + self.patch_size[0] > image.shape[0]:
                    x = image.shape[0] - self.patch_size[0]
                if y + self.patch_size[1] > image.shape[1]:
                    y = image.shape[1] - self.patch_size[1]
                image_patch = image[x:x + self.patch_size[0], y:y + self.patch_size[1], :]/255.0
                image_patch = normalize(image_patch, between=[-1,1])
                image_patch = np.expand_dims(image_patch,0)
                prob_op, class_op = self.sess.run(self.evaluation_ops, feed_dict=self.feed_dict(image_patch, False))

                heat_map[x:x + self.patch_size[0], y:y + self.patch_size[1], 0].fill(prob_op[0][0])
                heat_map[x:x + self.patch_size[0], y:y + self.patch_size[1], 1].fill(prob_op[0][1])

        return heat_map
