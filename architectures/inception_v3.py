from net.layers import atrous_conv, atrous_pool, conv, pool
from net.ops import concat

class Net(object):
    def __call__(self, inp, n_classes, train, *args, **kwargs):
        # [None , 224 ,224 ,3]

        conv1 = atrous_conv(inp, (3, 3), 16,
                            atrous_stride=(2, 2),
                            strides=(1, 1),
                            weight_initializer={"name": "xavier"},
                            padding="VALID", activation_func={"name": "crelu"}, name="atrous_conv1", print_shape=True)
        # [None, 224 ,224, 64 ]
        conv2 = atrous_conv(conv1, (3, 3), 16,
                            atrous_stride=(2, 2),
                            strides=(1, 1),
                            weight_initializer={"name": "xavier"},
                            padding="VALID", activation_func={"name": "crelu"}, batch_norm_func={"phase_train": train},
                            name="atrous_conv2", print_shape=True)
        # [ None, 224, 224, 64]
        conv3 = atrous_conv(conv2, (3, 3), 32,
                            atrous_stride=(2, 2),
                            strides=(1, 1),
                            weight_initializer={"name": "xavier"},
                            padding="VALID", activation_func={"name": "crelu"}, batch_norm_func={"phase_train": train},
                            name="atrous_conv3", print_shape=True)
        # [ None, 224, 224, 128]
        pool1 = atrous_pool(conv3, (3, 3), stride=(1, 1), atrous_stride=(2, 2), pooling_type="MAX", name="atrous_pool1",
                            print_shape=True)
        # [None,111, 111,128]
        conv4 = atrous_conv(pool1, (3, 3), 40,
                            atrous_stride=(2, 2),
                            strides=(1, 1),
                            weight_initializer={"name": "xavier"},
                            padding="VALID", activation_func={"name": "crelu"}, batch_norm_func={"phase_train": train},
                            name="atrous_conv4", print_shape=True)

        pool2 = pool(conv4, (3, 3), stride=(2, 2), pooling_type="MAX", name="pool2",
                            print_shape=True)

        branch1x1 = atrous_conv(pool2, (1, 1), 32,
                                atrous_stride=(2, 2),
                                strides=(1, 1),
                                weight_initializer={"name": "xavier"},
                                padding="SAME", activation_func={"name": "crelu"},
                                batch_norm_func={"phase_train": train},
                                name="atrous_branch1x1", print_shape=True)
        branch5x5 = atrous_conv(pool2, (1, 1), 24,
                                atrous_stride=(2, 2),
                                strides=(1, 1),
                                weight_initializer={"name": "xavier"},
                                padding="SAME", activation_func={"name": "crelu"},
                                batch_norm_func={"phase_train": train},
                                name="atrous_branch5x5", print_shape=True)
        branch1x1_5x5 = atrous_conv(branch5x5, (5, 5), 32,
                                    atrous_stride=(2, 2),
                                    strides=(1, 1),
                                    weight_initializer={"name": "xavier"},
                                    padding="SAME", activation_func={"name": "crelu"},
                                    batch_norm_func={"phase_train": train},
                                    name="atrous_branch1x1_5x5", print_shape=True)

        branch3x3 = atrous_conv(pool2, (1, 1), 32,
                                atrous_stride=(2, 2),
                                strides=(1, 1),
                                weight_initializer={"name": "xavier"},
                                padding="SAME", activation_func={"name": "crelu"},
                                batch_norm_func={"phase_train": train},
                                name="atrous_branch3x3", print_shape=True)
        branchd3x3 = atrous_conv(branch3x3, (3, 3), 48,
                                 atrous_stride=(2, 2),
                                 strides=(1, 1),
                                 weight_initializer={"name": "xavier"},
                                 padding="SAME", activation_func={"name": "crelu"},
                                 batch_norm_func={"phase_train": train},
                                 name="atrous_branchd3x3", print_shape=True)
        branchd3x3 = atrous_conv(branchd3x3, (3, 3), 48,
                                 atrous_stride=(2, 2),
                                 strides=(1, 1),
                                 weight_initializer={"name": "xavier"},
                                 padding="SAME", activation_func={"name": "crelu"},
                                 batch_norm_func={"phase_train": train},
                                 name="atrous_branchd3x3_1", print_shape=True)
        pool3 = pool(pool2, (3, 3), stride=(1,1), padding="SAME", pooling_type="AVG", name="atrous_pool3",
                            print_shape=True)

        branchpool = atrous_conv(pool3, (1, 1), 16,
                                 atrous_stride=(2, 2),
                                 strides=(1, 1),
                                 weight_initializer={"name": "xavier"},
                                 padding="SAME", activation_func={"name": "crelu"},
                                 batch_norm_func={"phase_train": train},
                                 name="atrous_branchpool", print_shape=True)

        net = concat(axis=3, values=[branch1x1, branch1x1_5x5, branchd3x3, branchpool], name="net_concat_1")

        branch1x1_1 = atrous_conv(net, (1, 1), 32,
                                 atrous_stride=(2, 2),
                                 strides=(1, 1),
                                 weight_initializer={"name": "xavier"},
                                 padding="SAME", activation_func={"name": "crelu"},
                                 batch_norm_func={"phase_train": train},
                                 name="atrous_branch1x1_1", print_shape=True)
        branch5x5_1 = atrous_conv(net, (1, 1), 24,
                                 atrous_stride=(2, 2),
                                 strides=(1, 1),
                                 weight_initializer={"name": "xavier"},
                                 padding="SAME", activation_func={"name": "crelu"},
                                 batch_norm_func={"phase_train": train},
                                 name="atrous_branch5x5_1", print_shape=True)
        branch5x5_1d = atrous_conv(branch5x5_1, (5, 5), 16,
                                 atrous_stride=(2, 2),
                                 strides=(1, 1),
                                 weight_initializer={"name": "xavier"},
                                 padding="SAME", activation_func={"name": "crelu"},
                                 batch_norm_func={"phase_train": train},
                                 name="atrous_branch5x5_1d", print_shape=True)
        branch5x5_1 = atrous_conv(net, (1, 1), 24,
                                  atrous_stride=(2, 2),
                                  strides=(1, 1),
                                  weight_initializer={"name": "xavier"},
                                  padding="SAME", activation_func={"name": "crelu"},
                                  batch_norm_func={"phase_train": train},
                                  name="atrous_branch5x5_1e", print_shape=True)

        branch5x5_13d = atrous_conv(net, (1, 1), 32,
                               atrous_stride=(2, 2),
                               strides=(1, 1),
                               weight_initializer={"name": "xavier"},
                               padding="SAME", activation_func={"name": "crelu"},
                               batch_norm_func={"phase_train": train},
                               name="atrous_branch5x5_13d", print_shape=True)
        branch5x5_13d = atrous_conv(branch5x5_13d, (3, 3), 48,
                                  atrous_stride=(2, 2),
                                  strides=(1, 1),
                                  weight_initializer={"name": "xavier"},
                                  padding="SAME", activation_func={"name": "crelu"},
                                  batch_norm_func={"phase_train": train},
                                  name="atrous_branch5x5_13e", print_shape=True)

        branch5x5_13d = atrous_conv(branch5x5_13d, (3, 3), 48,
                               atrous_stride=(2, 2),
                               strides=(1, 1),
                               weight_initializer={"name": "xavier"},
                               padding="SAME", activation_func={"name": "crelu"},
                               batch_norm_func={"phase_train": train},
                               name="atrous_branch5x5_13f", print_shape=True)

        pool4 = pool(net, (3, 3), stride=(1, 1), padding="SAME", pooling_type="AVG", name="atrous_pool4",
                     print_shape=True)

        branchpool = atrous_conv(pool4, (1, 1), 32,
                             atrous_stride=(2, 2),
                             strides=(1, 1),
                             weight_initializer={"name": "xavier"},
                             padding="SAME", activation_func={"name": "crelu"},
                             batch_norm_func={"phase_train": train},
                             name="atrous_branchpool_1", print_shape=True)
        net_2 = concat(axis=3, values=[branch1x1, branch5x5_1d, branch5x5_13d, branchpool], name="net_concat_2")

        net_2 = pool(net_2, (2, 2), stride=(2, 2), padding="VALID", pooling_type="MAX", name="atrous_pool5",
                     print_shape=True)
        # with tf.variable_scope('mixed_17x17x768a'):
        #     with tf.variable_scope('branch3x3'):
        #         branch3x3 = ops.conv2d(net, 384, [3, 3], stride=2, padding='VALID')
        #     with tf.variable_scope('branch3x3dbl'):
        #         branch3x3dbl = ops.conv2d(net, 64, [1, 1])
        #         branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
        #         branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],
        #                                   stride=2, padding='VALID')
        #     with tf.variable_scope('branch_pool'):
        #         branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
        #     net = tf.concat(axis=3, values=[branch3x3, branch3x3dbl, branch_pool])

        branch3x3_1 = atrous_conv(net_2, (3, 3), 192,
                             atrous_stride=(2, 2),
                             strides=(1, 1),
                             weight_initializer={"name": "xavier"},
                             padding="VALID", activation_func={"name": "crelu"},
                             batch_norm_func={"phase_train": train},
                             name="atrous_branch_3x3_1", print_shape=True)
        branch3x3db1 = atrous_conv(net_2, (1, 1), 64,
                             atrous_stride=(2, 2),
                             strides=(1, 1),
                             weight_initializer={"name": "xavier"},
                             padding="SAME", activation_func={"name": "crelu"},
                             batch_norm_func={"phase_train": train},
                             name="atrous_branch_3x3db11", print_shape=True)
        branch3x3db1 = atrous_conv(branch3x3db1, (3, 3), 48,
                                   atrous_stride=(2, 2),
                                   strides=(1, 1),
                                   weight_initializer={"name": "xavier"},
                                   padding="SAME", activation_func={"name": "crelu"},
                                   batch_norm_func={"phase_train": train},
                                   name="atrous_branch_3x3db2", print_shape=True)
        branch3x3db1 = atrous_conv(branch3x3db1, (3, 3), 48,
                                   atrous_stride=(2, 2),
                                   strides=(1, 1),
                                   weight_initializer={"name": "xavier"},
                                   padding="VALID", activation_func={"name": "crelu"},
                                   batch_norm_func={"phase_train": train},
                                   name="atrous_branch_3x3db3", print_shape=True)

        branch_pool_6 = pool(net_2, (2, 2), stride=(1, 1), padding="SAME", pooling_type="MAX", name="pool6",
                     print_shape=True)
        net_3 = concat(axis=3,values=[branch3x3_1,branch3x3db1,branch_pool_6],name="net_3")


        #     end_points['mixed_17x17x768a'] = net
        #     # mixed4: 17 x 17 x 768.
        # with tf.variable_scope('mixed_17x17x768b'):
        #     with tf.variable_scope('branch1x1'):
        #         branch1x1 = ops.conv2d(net, 192, [1, 1])
        #     with tf.variable_scope('branch7x7'):
        #         branch7x7 = ops.conv2d(net, 128, [1, 1])
        #         branch7x7 = ops.conv2d(branch7x7, 128, [1, 7])
        #         branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
        #     with tf.variable_scope('branch7x7dbl'):
        #         branch7x7dbl = ops.conv2d(net, 128, [1, 1])
        #         branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
        #         branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [1, 7])
        #         branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
        #         branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
        #     with tf.variable_scope('branch_pool'):
        #         branch_pool = ops.avg_pool(net, [3, 3])
        #         branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
        #     net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
        #     end_points['mixed_17x17x768b'] = net
        #     # mixed_5: 17 x 17 x 768.
        # with tf.variable_scope('mixed_17x17x768c'):
        #     with tf.variable_scope('branch1x1'):
        #         branch1x1 = ops.conv2d(net, 192, [1, 1])
        #     with tf.variable_scope('branch7x7'):
        #         branch7x7 = ops.conv2d(net, 160, [1, 1])
        #         branch7x7 = ops.conv2d(branch7x7, 160, [1, 7])
        #         branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
        #     with tf.variable_scope('branch7x7dbl'):
        #         branch7x7dbl = ops.conv2d(net, 160, [1, 1])
        #         branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
        #         branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [1, 7])
        #         branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
        #         branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
        #     with tf.variable_scope('branch_pool'):
        #         branch_pool = ops.avg_pool(net, [3, 3])
        #         branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
        #     net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
        #     end_points['mixed_17x17x768c'] = net


        return conv1
