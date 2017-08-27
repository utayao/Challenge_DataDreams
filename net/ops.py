import tensorflow as tf
import tf_utils


def pixel_wise_softmax(logits):
    exp_map = tf.exp(logits)
    sum_exp = tf.reduce_sum(exp_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(logits)[3]]))
    return tf.div(exp_map, tensor_sum_exp)


def merge(layer1, layer2, axis=-1, name="merge"):
    name = tf_utils.get_unique_name(name)
    with tf.name_scope(name):
        return tf.concat([layer1, layer2], axis=axis, name="merge_layers")


def batch_norm(inp, phase_train, name="batch_norm"):
    n_out = tf_utils.get_inp_shape(inp)[-1]

    with tf.variable_scope(name):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                       name="beta", trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name="gamma", trainable=True)
        batch_mean, batch_var = tf.nn.moments(inp, [0, 1, 2], name="moments")
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inp, mean, var, beta, gamma, 1e-3)

        return normed


def dropout(inp, keep_prob,name):
    name = tf_utils.get_unique_name(name)
    with tf.name_scope(name):
        dropout = tf.nn.dropout(inp,keep_prob=keep_prob,name="dropout")
    return dropout
