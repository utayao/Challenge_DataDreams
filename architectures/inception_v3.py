from net.layers import atrous_conv, atrous_pool, conv, pool


class Net(object):
    def __call__(self, inp,n_classes,train, *args, **kwargs):
        # [None , 224 ,224 ,3]

        conv1 = atrous_conv(inp, (3, 3), 32,
                            atrous_stride=(2, 2),
                            strides=(1,1),
                            weight_initializer={"name": "xavier"},
                            padding="VALID", activation_func={"name": "crelu"}, name="atrous_conv1",print_shape=True)
        # [None, 224 ,224, 64 ]
        conv2 = atrous_conv(conv1, (3, 3), 32,
                            atrous_stride=(2, 2),
                            strides=(1,1),
                            weight_initializer={"name": "xavier"},
                            padding="VALID", activation_func={"name": "crelu"}, name="atrous_conv2",print_shape=True)
        # [ None, 224, 224, 64]
        conv3 = atrous_conv(conv2, (3, 3), 64,
                            atrous_stride=(2, 2),
                            strides=(1, 1),
                            weight_initializer={"name": "xavier"},
                            padding="VALID", activation_func={"name": "crelu"}, name="atrous_conv3",print_shape=True)
        # [ None, 224, 224, 128]
        pool1 = atrous_pool(conv3,(3,3),stride=(1,1),atrous_stride=(2,2),pooling_type="MAX",name="atrous_pool1",print_shape=True)
        # [None,111, 111,128]
        conv4 = atrous_conv(pool1, (3, 3), 80,
                            atrous_stride=(2, 2),
                            strides=(1, 1),
                            weight_initializer={"name": "xavier"},
                            padding="VALID", activation_func={"name": "crelu"}, name="atrous_conv4", print_shape=True)

        pool2 = atrous_pool(conv4,(3,3),stride=(1,1),atrous_stride=(2,2),pooling_type="MAX",name="atrous_pool2",print_shape=True)


        return conv1