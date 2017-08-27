""" weights initialization """

import tensorflow as tf
from numpy import sqrt
from tf_utils import get_from_module


def random_normal(**weights_params):
    mean = weights_params.get("mean", 0.0)
    stddev = weights_params.get("stddev", 1.0)
    seed = weights_params.get("seed", 12345)
    dtype = weights_params.get("dtype", tf.float32)
    return tf.random_normal_initializer(
        mean=mean,
        stddev=stddev,
        seed=seed,
        dtype=dtype
    )


def xavier(**weight_params):
    return tf.contrib.layers.xavier_initializer()

def kamming_he(**weights_params):
    assert "in_shape" in weights_params, "Please specify the incoming shape"
    weights_params["stddev"] = sqrt((2.0/weights_params["in_shape"]))
    return random_normal(**weights_params)

def get_weights(weights_initializer_dict, kernel_size, name):
    """
    get weights for mentioned dict
    :param weights_initializer_dict: weights dict
    :param kernel_size: kernel size
    :param name: weights name
    :return: tensor
    """
    with tf.variable_scope(name):
        var = tf.get_variable(
            "weights",
            shape=kernel_size,
            initializer=get(weights_initializer_dict))

    return var


def get_bias(bias_initializer_dict, num_output, name):
    """
    get bias for mentioned dict
    :param bias_initializer_dict: bias dict
    :param num_output: vector size
    :param name: bias name
    :return: tensor
    """
    with tf.variable_scope(name):
        bias_var = tf.get_variable(
            "bias", [num_output],
            initializer=get(bias_initializer_dict))
    return bias_var


def get(initializer_dict):
    return get_from_module(
        initializer_dict["name"],
        type_name="initializations",
        module_obj=globals(),
        module_params=initializer_dict
    )