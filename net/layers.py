""" Deep learning layers implemented in tensorflow """

import tensorflow as tf

import tf_utils
import tf_activations
import tf_initializations
from ops import batch_norm, dropout


def atrous_conv(
        inp, kernel_shape,
        num_output, weight_initializer={
            "name": "random_normal",
            "mean": 0.0,
            "stddev": 1.0
        },
        bias_initializer={
            "name": "random_normal",
            "mean": 0.0,
            "stddev": 1.0
        },
        atrous_stride=[1, 1],
        strides=None,
        padding="VALID",
        activation_func=None,
        batch_norm_func=None,
        dropout_func=None,
        print_shape=False,
        name="atrous_conv_sk"
):
    name = tf_utils.get_unique_name(name)
    inp_shape = tf_utils.get_inp_shape(inp)
    atrous_height, atrous_width = atrous_stride
    with tf.name_scope(name):
        kernel_shape = list(kernel_shape) + [inp_shape[-1], num_output]
        weights = tf_initializations.get_weights(weight_initializer, kernel_shape, name)
        bias = tf_initializations.get_bias(bias_initializer, num_output, name)
        assert any([isinstance(weights, object), isinstance(bias, object)]), 'weights or bias can not be None'
        conv_op = tf.nn.convolution(inp, weights, padding, strides=strides, dilation_rate=[atrous_height, atrous_width],
                                    name='conv')
        bias_add_op = tf.nn.bias_add(conv_op, bias, name='bias_add')
    if activation_func:
        bias_add_op = tf_activations.get(activation_func)(bias_add_op)
    assert isinstance(bias_add_op, object)
    if print_shape:
        print "Layer type: {} shape: {}".format(name, tf_utils.get_inp_shape(bias_add_op))
    if batch_norm_func:
        assert "phase_train" in batch_norm_func, "phase train should be present in dict"
        bias_add_op = batch_norm(bias_add_op, batch_norm_func["phase_train"], "batch_norm")
    if dropout_func:
        assert "keep_prob" in dropout_func, "keep_prob should be present in dict"

        bias_add_op = dropout(bias_add_op, keep_prob=dropout_func["keep_prob"], name=name)

    return bias_add_op


def conv(
        inp, kernel_shape,
        num_output, weight_initializer={
            "name": "random_normal",
            "mean": 0.0,
            "stddev": 1.0
        },
        bias_initializer={
            "name": "random_normal",
            "mean": 0.0,
            "stddev": 1.0
        },
        strides=None,
        padding="VALID",
        activation_func=None,
        dropout_func=None,
        batch_norm_func=None,
        print_shape=False,
        name="conv_sk"
):
    """
    Conv doc string
    :param inp:
    :param kernel_shape:
    :param num_output:
    :param weight_initializer:
    :param bias_initializer:
    :param padding:
    :param activation_func:
    :param name:
    :return:
    """
    name = tf_utils.get_unique_name(name)
    inp_shape = tf_utils.get_inp_shape(inp)
    with tf.name_scope(name):
        kernel_shape = list(kernel_shape) + [inp_shape[-1], num_output]
        weights = tf_initializations.get_weights(weight_initializer, kernel_shape, name)
        bias = tf_initializations.get_bias(bias_initializer, num_output, name)
        assert any([isinstance(weights, object), isinstance(bias, object)]), 'weights or bias can not be None'
        conv_op = tf.nn.convolution(inp, weights, padding, strides=strides, name='conv')
        bias_add_op = tf.nn.bias_add(conv_op, bias, name='bias_add')
    if activation_func:
        bias_add_op = tf_activations.get(activation_func)(bias_add_op)
    assert isinstance(bias_add_op, object)
    if print_shape:
        print "Layer type: {} shape: {}".format(name, tf_utils.get_inp_shape(bias_add_op))

    if batch_norm_func:
        assert "phase_train" in batch_norm_func, "phase train should be present in dict"
        bias_add_op = batch_norm(bias_add_op, batch_norm_func["phase_train"], "batch_norm")
    if dropout_func:
        assert "keep_prob" in dropout_func, "keep_prob should be present in dict"
        bias_add_op = dropout(bias_add_op, keep_prob=dropout_func["keep_prob"], name=name)

    return bias_add_op


def atrous_pool(
        inp,
        kernel_shape=(2, 2),
        stride=(1, 1),
        atrous_stride=(2, 2),
        pooling_type="MAX",
        padding="VALID",
        print_shape=False,
        name="atrous_pool"
):
    """
    TODO: pool doc string
    :param inp:
    :param kernel_shape:
    :param stride:
    :param pooling_type:
    :param padding:
    :param name:
    :return:
    """
    name = tf_utils.get_unique_name(name)
    with tf.name_scope(name):
        pool_op = tf.nn.pool(
            inp,
            kernel_shape,
            pooling_type,
            padding,
            strides=stride,
            dilation_rate=atrous_stride,
            name=name
        )
        if print_shape:
            print "Layer type: {} shape: {}".format(name, tf_utils.get_inp_shape(pool_op))

        return pool_op


def pool(
        inp,
        kernel_shape=(2, 2),
        stride=(2, 2),
        pooling_type="MAX",
        padding="VALID",
        print_shape=False,
        name="pool"
):
    """
    TODO: pool doc string
    :param inp:
    :param kernel_shape:
    :param stride:
    :param pooling_type:
    :param padding:
    :param name:
    :return:
    """
    name = tf_utils.get_unique_name(name)
    with tf.name_scope(name):
        pool_op = tf.nn.pool(
            inp,
            kernel_shape,
            pooling_type,
            padding,
            strides=stride,
            name=name
        )
        if print_shape:
            print "Layer type: {} shape: {}".format(name, tf_utils.get_inp_shape(pool_op))

        return pool_op


def deconv(inp, weights, stride, padding="VALID", name="deonv"):
    """
    Deconvolution
    :param inp:
    :param weights:
    :param stride:
    :param name:
    :return:
    """
    name = tf_utils.get_unique_name(name)
    inp_shape = tf_utils.get_inp_shape(inp)
    output_shape = tf.stack([inp_shape[0], inp_shape[1] * 2, inp_shape[2] * 2, inp_shape[3] // 2])
    with tf.name_scope(name):
        return tf.nn.conv2d_transpose(
            inp,
            weights,
            output_shape,
            strides=[1, stride, stride, 1],
            padding=padding
        )


def upsampling(inp, size=(2, 2), name="upsampling"):
    """

    :param inp:
    :param size:
    :return:
    """
    H, W, _ = inp.get_shape().as_list()[1:]
    H_multiplier = H * size[0]
    W_multiplier = W * size[1]
    return tf.image.resize_nearest_neighbor(inp, size=(H_multiplier, W_multiplier), name=name)

def fc(
        inp,
        num_output, weight_initializer={
            "name": "random_normal",
            "mean": 0.0,
            "stddev": 1.0,
            "restore":False
        },
        bias_initializer={
            "name": "random_normal",
            "mean": 0.0,
            "stddev": 1.0,
            "restore":False
        },
        activation_func=None,
        dropout_func=None,
        batch_norm_func=None,
        restore=False,
        print_shape=False,
        name="fc"
):
    name = tf_utils.get_unique_name(name)
    inp_shape = tf_utils.get_inp_shape(inp)
    with tf.name_scope(name):
        kernel_shape = [inp_shape[-1], num_output]
        weights = tf_initializations.get_weights(weight_initializer, kernel_shape, name)
        bias = tf_initializations.get_bias(bias_initializer, num_output, name)
        assert any([isinstance(weights, object), isinstance(bias, object)]), 'weights or bias can not be None'
        bias_add_op = tf.nn.xw_plus_b(inp, weights,bias,name='fc')
    if activation_func:
        bias_add_op = tf_activations.get(activation_func)(bias_add_op)
    assert isinstance(bias_add_op, object)
    if print_shape:
        print "Layer type: {} shape: {}".format(name, tf_utils.get_inp_shape(bias_add_op))

    if batch_norm_func:
        assert "phase_train" in batch_norm_func, "phase train should be present in dict"
        bias_add_op = batch_norm(bias_add_op, batch_norm_func["phase_train"], "batch_norm")
    if dropout_func:
        assert "keep_prob" in dropout_func, "keep_prob should be present in dict"
        bias_add_op = dropout(bias_add_op, keep_prob=dropout_func["keep_prob"], name=name)

    return bias_add_op