import tf_args
from tensorflow.python.platform import gfile
import tensorflow as tf
from trainer import UNetTrainer
#from architecture import Unet

FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    if FLAGS.debug:
        from tensorflow.python import debug as tf_debug
        sess = tf.InteractiveSession()
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    else:
        sess = None

    data_augmentation = None
    gfile.MakeDirs(FLAGS.train_dir)
    trainer = UNetTrainer(None, FLAGS.data_dir, FLAGS.train_dir, data_augmentation=data_augmentation, cv=FLAGS.cv,
                          subset=FLAGS.subset,
                          image_resize=(FLAGS.image_resize, FLAGS.image_resize),
                          batch_size=FLAGS.batch_size,
                          normalize=FLAGS.normalize)
    trainer.build()
    trainer.train(max_iter=FLAGS.iter, sess=sess)


if __name__ == "__main__":
    tf.app.run()