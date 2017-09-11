"""
Build tensorflow flags
"""

import tensorflow as tf

DISPLAY_ITERS = 10
EVAL_ITERS = 50
SAVE_ITERS = 150
EVAL_COUNT = 20
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997
MOVING_AVERAGE_DECAY = 0.9999
IMAGES_MEAN_PATH = "../data"
IMAGES_STD_PATH = "../data"
TOWER_NAME = 'tower'
UPDATE_OPS_COLLECTION  = '_update_ops_'

# File I/O
tf.app.flags.DEFINE_string("data_dir", "/home/ashwin/Challenge_DataDreams/data", "path where the data is located")
tf.app.flags.DEFINE_string("train_dir", "results", "path to store train model")
tf.app.flags.DEFINE_string('state_file','/home/ashwin/Challenge_DataDreams/phase_1/state.json','path to store states')
# Model hyper parameters
tf.app.flags.DEFINE_integer("batch_size", 16, "Define batch size")
tf.app.flags.DEFINE_integer("iter", 10000, "Iterations to train the data. Note that this is not epoch")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
tf.app.flags.DEFINE_integer("MAX_GRADIENT_NORM", 50, "Maximum Gradient Norm")

# data hyper parameters
tf.app.flags.DEFINE_integer("cv", 2, "cross validation")
tf.app.flags.DEFINE_boolean("subset", True, "subset of data")
tf.app.flags.DEFINE_boolean("normalize", False, "Normalize the data")
tf.app.flags.DEFINE_integer("image_resize", 224, "cross validation")
tf.app.flags.DEFINE_boolean("plot", False, "plot the images")
tf.app.flags.DEFINE_string("file_path", None, "file path")
tf.app.flags.DEFINE_string("dir_path", None, "directory path")
tf.app.flags.DEFINE_string("model_path", "/home/ashwin/carvana/unet/results/checkpoints/cv_0", "model path")



tf.app.flags.DEFINE_boolean("debug", False, "debugger")
