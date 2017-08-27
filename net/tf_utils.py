""" utils for implementing tf """

import tensorflow as tf
import numpy as np
import six


def get_unique_name(name):
    """
    find unique name
    :param name: name of the tensor
    :return: unique name
    """
    counter = -1

    def __unique_name_helper(variable_name_scopes, name):
        global counter
        if name in variable_name_scopes:
            counter += 1
            name += "_" + str(counter)
            return __unique_name_helper(variable_name_scopes, name)
        return name

    variable_name_scopes = [n.name for n in tf.global_variables()]
    return __unique_name_helper(variable_name_scopes, name)


def get_inp_shape(inp):
    """
    find shape of the tensor
    :param inp:
    :return: shape (B, H, W, c)
    """
    if isinstance(inp, tf.Tensor):
        return inp.get_shape().as_list()
    elif type(inp) in [np.array, np.ndarray, list, tuple]:
        return np.shape(inp)
    else:
        raise Exception("Invalid input shape.")


def get_from_module(identifier,
                    type_name,
                    module_obj=globals(),
                    module_params=None
                    ):
    """
    get the respective function.
    :param identifier: function name as string
    :param type_name: type of module
    :param module_obj: array of functions tpically globals or locals
    :param module_params: function parameter
    :return: function
    """
    if isinstance(identifier, six.string_types):
        res = module_obj.get(identifier)
        if not res:
            res = module_obj.get(identifier.lower())
            if not res:
                raise Exception('Invalid ' + str(type_name))
        if not module_params:
            return res()
        elif module_params:
            return res(**module_params)
        else:
            return res
    return identifier


def resize_tf_images(images, size, name="resize"):

    return tf.image.resize_nearest_neighbor(images, size=size, name=name)