""" activations """

import tensorflow as tf
from tf_utils import get_from_module


def relu(**params):
    return tf.nn.relu

def softmax(**params):

    return tf.nn.softmax

def crelu(**params):
    return tf.nn.crelu

def get(initializer_dict):

    return get_from_module(
                    initializer_dict["name"],
                    type_name="activations",
                    module_obj=globals(),
                    module_params=initializer_dict
                    )
