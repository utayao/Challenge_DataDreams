from net.layers import atrous_conv, atrous_pool,pool


class Net(object):
    def __call__(self, inp, n_classes, train):

        conv1 = atrous_conv(inp, (5, 5), 48,
                                 atrous_stride=(2, 2),
                                 strides=(1, 1),
                                 weight_initializer={"name": "xavier"},
                                 padding="VALID", activation_func={"name":"crelu"},
                                 batch_norm_func=None,
                                 name="atrous_conv1", print_shape=True)
        pool1 = pool(conv1, (2, 2), stride=(1, 1), pooling_type="MAX", name="pool1",
                            print_shape=True)
        conv2 = atrous_conv(pool1, (5, 5), 25,
                            atrous_stride=(2, 2),
                            strides=(1, 1),
                            weight_initializer={"name": "xavier"},
                            padding="VALID", activation_func={"name":"crelu"},
                            batch_norm_func={"phase_train": train},
                            name="atrous_conv2", print_shape=True)
        pool2 = atrous_pool(conv2, (3, 3), stride=(1, 1), atrous_stride=(2, 2), pooling_type="MAX", name="atrous_pool3",
                            print_shape=True)
        conv2 = atrous_conv(pool2, (5, 5), 250,
                            atrous_stride=(2, 2),
                            strides=(1, 1),
                            weight_initializer={"name": "xavier"},
                            padding="VALID", activation_func={"name": "crelu"},
                            batch_norm_func={"phase_train": train},
                            name="atrous_conv2", print_shape=True)

        return conv1