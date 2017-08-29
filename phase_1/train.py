import tensorflow as tf
from tensorflow.python.platform import gfile

from step_1.trainer import NetTrainer
from architectures.inception_v3_slim import Net

FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    if FLAGS.debug:
        from tensorflow.python import debug as tf_debug
        sess = tf.InteractiveSession()
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    else:
        sess = None

    cancer_data_augmentation = {

    }
    non_cancer_data_augmentation = {

    }
    gfile.MakeDirs(FLAGS.train_dir)
    trainer = NetTrainer(Net(), FLAGS.data_dir, FLAGS.train_dir, cancer_data_augmentation=cancer_data_augmentation,
                          non_cancer_data_augmentation=non_cancer_data_augmentation, cv=FLAGS.cv,
                          subset=FLAGS.subset,
                          image_resize=(FLAGS.image_resize, FLAGS.image_resize),
                          batch_size=FLAGS.batch_size,
                          normalize=FLAGS.normalize)
    trainer.build()
    trainer.train(max_iter=FLAGS.iter, sess=sess)


if __name__ == "__main__":
    tf.app.run()
