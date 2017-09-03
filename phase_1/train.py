import tensorflow as tf
from tensorflow.python.platform import gfile

from step_1.trainer import NetTrainer

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
        'rotate_image': {'angle':[1,5]},
        'flipud': True,
        'fliplr': True,
        'adjust_brightness': True,
        'adjust_hue':True,
        'adjust_saturation':True


    }
    non_cancer_data_augmentation = cancer_data_augmentation.copy()

    gfile.MakeDirs(FLAGS.train_dir)
    trainer = NetTrainer(FLAGS.data_dir, FLAGS.train_dir, cancer_data_augmentation=cancer_data_augmentation,
                          non_cancer_data_augmentation=non_cancer_data_augmentation, cv=FLAGS.cv,
                          subset=FLAGS.subset,
                          image_resize=(FLAGS.image_resize, FLAGS.image_resize),
                          batch_size=FLAGS.batch_size,
                          normalize=FLAGS.normalize)
    trainer.build()
    trainer.train(max_iter=FLAGS.iter, sess=sess)


if __name__ == "__main__":
    tf.app.run()
