import tensorflow as tf
import numpy as np
import pdb
import utils
import os
import dataPrep as dp
from sklearn.utils import shuffle
from sklearn import model_selection
import skimage.io as io
from data_gen3 import TrainGenrator

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("lr", 0.0001, "learning rate")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("iters", 1000000, "epochs")
flags.DEFINE_integer("num_classes", 2, "number of classes")
flags.DEFINE_string("summary_path", "summary_path", "path to save summary")
flags.DEFINE_string("model_path", "model_path", "path to save model")
flags.DEFINE_string("checkpoint_path", "checkpoints", "checkpoint path")

# flags.DEFINE_string("epoch",15,"epochs")
flags.DEFINE_bool("pre_trained", False, "pre trained weights")
flags.DEFINE_string("weight_file", "Inception_weights.npz", "weights")
flags.DEFINE_string("model_name", "googlenet", "model name")
flags.DEFINE_string('data_path', '/home/ashwin/Challenge_DataDream/data/patches_new', 'data path')

x_train, y_train = dp.load_data(FLAGS.train_path)
# x_train = np.array(x_train)
# y_train = np.array(y_train)
# print x_train.shape
# x_train = np.swapaxes(x_train,1,2)
# io.imshow(x_train[0,...])
# x_train = x_train.astype(np.float32)
# mean = np.mean(x_train,axis=0)
# std = np.std(x_train,axis=0)
# x_train = x_train - mean
# x_train/=std
# pdb.set_trace()
fold = 1
params = {
    "rotate_image": {'start': 0, 'end': 20},

    # "shear":(0,0),
    # "translate":(-4,4),
    "flipud": True,
    "fliplr": True,
    # "stretch": (1/1.1,1.1)
}
cvs = 5
# if not os.path.exists('splits_subset_{}_cvs_{}.txt'.format(FLAGS.subset,cvs)):
#     _splits = model_selection.ShuffleSplit(n_splits=cvs, test_size=0.1)
#     indicies = [(train_index, test_index) for train_index, test_index in
#                      _splits.split(x_train)]
#     utils.save_obj(indicies, 'splits_subset_{}_cvs_{}.txt'.format(FLAGS.subset,cvs))
# else:
#     print 'cross validation avaliable'
#     indicies = utils.load_obj('splits_subset_{}_cvs_{}.txt'.format(FLAGS.subset,cvs))


for cv in range(cvs):

    # for train_index,valid_index in indicies[cv]:
    # x_traindata = [indices[i] for i in train_index]
    # y_traindata = [indices[i] for i in valid_index]
    # train_index, valid_index = indicies[cv][0], indicies[cv][1]
    # x_traindata, y_traindata = x_train[train_index], y_train[train_index]
    # x_validata, y_validata = x_train[valid_index], y_train[valid_index]

    with tf.Graph().as_default(), tf.Session() as sess:
        # from architectures import VGG16
        import model

        # layers = VGG16.VGG16(FLAGS.num_classes)
        m = model.Model(cv, params)
        print "Training size: {} , Validation size: {}".format(x_traindata.shape[0], x_validata.shape[0])
        for epoch in range(FLAGS.epochs):
            # x_traindata, y_traindata = shuffle(x_traindata, y_traindata)
            m.fit(epoch,is_training=True)
            # print "Epoch: {} training_loss: {} training_Accuracy: {}".format(epoch,train_loss,train_acc)
            # x_validata, y_validata = shuffle(x_validata, y_validata)
            m.fit(epoch, x_validata, y_validata, False)

            # print "==> "
            dp.display(epoch,
                       fold=fold,
                       training_loss=train_loss,
                       validation_loss=valid_loss,
                       training_acc=train_acc,
                       validation_acc=valid_acc)
