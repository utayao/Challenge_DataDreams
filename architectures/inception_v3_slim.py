from net.layers import conv, pool, fc
from net.ops import concat, identity, flatten , dropout


class Net(object):
    def __call__(self, inp, n_classes, train):
        print inp.get_shape()
        conv0 = conv(
            inp, [3, 3],
            32, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(2, 2),
            padding="VALID",
            activation_func={"name": "crelu"},
            dropout_func=None,
            batch_norm_func=None,
            print_shape=True,
            name="conv0"
        )
        conv1 = conv(
            conv0, [3, 3],
            32, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="VALID",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="conv1"
        )
        conv2 = conv(
            conv1, [3, 3],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="conv2"
        )
        pool1 = pool(conv2, (3, 3), stride=(2, 2), pooling_type="MAX", name="pool1",
                     print_shape=True)
        conv3 = conv(
            pool1, [1, 1],
            80, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="VALID",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="conv3"
        )
        conv4 = conv(
            conv3, [3, 3],
            192, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="VALID",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="conv4"
        )
        pool2 = pool(conv4, (3, 3), stride=(2, 2), pooling_type="MAX", name="pool2",
                     print_shape=True)
        net = pool2

        branch1x1 = conv(
            net, [1, 1],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x1"
        )
        branch5x5 = conv(
            net, [1, 1],
            48, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch5x5"
        )
        branch5x5 = conv(
            branch5x5, [5, 5],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch5x5_1"
        )
        branch3x3db1 = conv(
            net, [1, 1],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db1"
        )
        branch3x3db1 = conv(
            branch3x3db1, [3, 3],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db1_1"
        )
        branch3x3db1 = conv(
            branch3x3db1, [3, 3],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db1_2"
        )
        branch_pool = pool(net, (3, 3), stride=(1, 1), padding="SAME", pooling_type="AVG", name="pool3",
                           print_shape=True)
        branch_pool = conv(
            branch_pool, [1, 1],
            32, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch_pool"
        )
        net = concat(axis=3, values=[branch1x1, branch5x5, branch3x3db1, branch_pool], name="concat1")

        branch1x1 = conv(
            net, [1, 1],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x1_1"
        )
        branch5x5 = conv(
            net, [1, 1],
            48, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch5x5_1-1"
        )
        branch5x5 = conv(
            branch5x5, [5, 5],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch5x5_1_1"
        )
        branch3x3db1 = conv(
            net, [1, 1],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db1_1-1"
        )
        branch3x3db1 = conv(
            branch3x3db1, [3, 3],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db1_1_1"
        )
        branch3x3db1 = conv(
            branch3x3db1, [3, 3],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db1_1_2"
        )
        branch_pool = pool(net, (3, 3), stride=(1, 1), padding="SAME", pooling_type="AVG", name="pool4",
                           print_shape=True)
        branch_pool = conv(
            branch_pool, [1, 1],
            32, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch_pool_1"
        )
        net = concat(axis=3, values=[branch1x1, branch5x5, branch3x3db1, branch_pool], name="concat2")
        branch1x1 = conv(
            net, [1, 1],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x1_1-2"
        )
        branch5x5 = conv(
            net, [1, 1],
            48, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch5x5_1-2"
        )
        branch5x5 = conv(
            branch5x5, [5, 5],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch5x5_1_1-2"
        )
        branch3x3db1 = conv(
            net, [1, 1],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db1_1-2"
        )
        branch3x3db1 = conv(
            branch3x3db1, [3, 3],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db1_1-2_2"
        )
        branch3x3db1 = conv(
            branch3x3db1, [3, 3],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db1_1_3"
        )
        branch_pool = pool(net, (3, 3), stride=(1, 1), padding="SAME", pooling_type="AVG", name="pool5",
                           print_shape=True)
        branch_pool = conv(
            branch_pool, [1, 1],
            32, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch_pool_2"
        )
        net = concat(axis=3, values=[branch1x1, branch5x5, branch3x3db1, branch_pool], name="concat3")
        branch3x3 = conv(
            net, [3, 3],
            384, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(2, 2),
            padding="VALID",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3_1-2"
        )

        branch3x3db1 = conv(
            net, [1, 1],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db1_1-3"
        )
        branch3x3db1 = conv(
            branch3x3db1, [3, 3],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db1_1-2_3"
        )
        branch3x3db1 = conv(
            branch3x3db1, [3, 3],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(2, 2),
            padding="VALID",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db1_1_4"
        )
        branch_pool = pool(net, (3, 3), stride=(2, 2), padding="VALID", pooling_type="MAX", name="pool6",
                           print_shape=True)

        net = concat(axis=3, values=[branch3x3, branch3x3db1, branch_pool], name="concat4")

        branch1x1 = conv(
            net, [1, 1],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x1_1-3"
        )

        branch7x7 = conv(
            net, [1, 1],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7-1"
        )
        branch7x7 = conv(
            branch7x7, [1, 7],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x7"
        )
        branch7x7 = conv(
            branch7x7, [7, 1],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x1"
        )
        branch7x7db1 = conv(
            net, [1, 1],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7db-1"
        )
        branch7x7db1 = conv(
            branch7x7db1, [7, 1],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x1db1"
        )
        branch7x7db1 = conv(
            branch7x7db1, [1, 7],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x7db1"
        )
        branch7x7db1 = conv(
            branch7x7db1, [7, 1],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x1db2"
        )
        branch7x7db1 = conv(
            branch7x7db1, [1, 7],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x7db2"
        )
        branch_pool = pool(net, (3, 3), stride=(1, 1), padding="SAME", pooling_type="AVG", name="pool7",
                           print_shape=True)
        branch_pool = conv(
            branch_pool, [1, 1],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7_pool"
        )
        net = concat(axis=3, values=[branch1x1, branch7x7, branch7x7db1, branch_pool], name="concat5")
        branch1x1 = conv(
            net, [1, 1],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x1_1-4"
        )

        branch7x7 = conv(
            net, [1, 1],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7-2"
        )
        branch7x7 = conv(
            branch7x7, [1, 7],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x7-2"
        )
        branch7x7 = conv(
            branch7x7, [7, 1],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x1-2"
        )
        branch7x7db1 = conv(
            net, [1, 1],
            80, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7db-2"
        )
        branch7x7db1 = conv(
            branch7x7db1, [7, 1],
            80, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x1db1-2"
        )
        branch7x7db1 = conv(
            branch7x7db1, [1, 7],
            80, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x7db1-2"
        )
        branch7x7db1 = conv(
            branch7x7db1, [7, 1],
            80, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x1db2-2"
        )
        branch7x7db1 = conv(
            branch7x7db1, [1, 7],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x7db2-2"
        )
        branch_pool = pool(net, (3, 3), stride=(1, 1), padding="SAME", pooling_type="AVG", name="pool8",
                           print_shape=True)
        branch_pool = conv(
            branch_pool, [1, 1],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7_pool-1"
        )
        net = concat(axis=3, values=[branch1x1, branch7x7, branch7x7db1, branch_pool], name="concat6")
        branch1x1 = conv(
            net, [1, 1],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x1_1-5"
        )

        branch7x7 = conv(
            net, [1, 1],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7-3"
        )
        branch7x7 = conv(
            branch7x7, [1, 7],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x7-3"
        )
        branch7x7 = conv(
            branch7x7, [7, 1],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x1-3"
        )
        branch7x7db1 = conv(
            net, [1, 1],
            80, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7db-3"
        )
        branch7x7db1 = conv(
            branch7x7db1, [7, 1],
            80, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x1db1-3"
        )
        branch7x7db1 = conv(
            branch7x7db1, [1, 7],
            80, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x7db1-3"
        )
        branch7x7db1 = conv(
            branch7x7db1, [7, 1],
            80, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x1db2-3"
        )
        branch7x7db1 = conv(
            branch7x7db1, [1, 7],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x7db2-3"
        )
        branch_pool = pool(net, (3, 3), stride=(1, 1), padding="SAME", pooling_type="AVG", name="pool9",
                           print_shape=True)
        branch_pool = conv(
            branch_pool, [1, 1],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7_pool-2"
        )
        net = concat(axis=3, values=[branch1x1, branch7x7, branch7x7db1, branch_pool], name="concat7")
        branch1x1 = conv(
            net, [1, 1],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x1_1-6"
        )

        branch7x7 = conv(
            net, [1, 1],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7-4"
        )
        branch7x7 = conv(
            branch7x7, [1, 7],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x7-4"
        )
        branch7x7 = conv(
            branch7x7, [7, 1],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x1-4"
        )
        branch7x7db1 = conv(
            net, [1, 1],
            80, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7db-4"
        )
        branch7x7db1 = conv(
            branch7x7db1, [7, 1],
            80, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x1db1-4"
        )
        branch7x7db1 = conv(
            branch7x7db1, [1, 7],
            80, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x7db1-4"
        )
        branch7x7db1 = conv(
            branch7x7db1, [7, 1],
            80, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x1db2-4"
        )
        branch7x7db1 = conv(
            branch7x7db1, [1, 7],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x7db2-4"
        )
        branch_pool = pool(net, (3, 3), stride=(1, 1), padding="SAME", pooling_type="AVG", name="pool10",
                           print_shape=True)
        branch_pool = conv(
            branch_pool, [1, 1],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7_pool-3"
        )
        net = concat(axis=3, values=[branch1x1, branch7x7, branch7x7db1, branch_pool], name="concat8")
        aux_logits = identity(net)
        aux_logits = pool(net, (5, 5), stride=(3, 3), padding="VALID", pooling_type="AVG", name="pool11",
                          print_shape=True)
        shape = aux_logits.get_shape()
        aux_logits = conv(
            aux_logits, shape[1:3],
            768, weight_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 0.01
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="VALID",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7_pool-4"
        )
        aux_logits = flatten(aux_logits)
        aux_logits = fc(aux_logits, n_classes, activation_func=None, weight_initializer={
            "name": "random_normal",
            "mean": 0.0,
            "stddev": 0.001,
            "restore": True
        },
                        bias_initializer={
                            "name": "random_normal",
                            "mean": 0.0,
                            "stddev": 1.0,
                            "restore": True
                        })
        branch3x3 = conv(
            net, [1, 1],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x1-6"
        )
        branch3x3 = conv(
            branch3x3, [3, 3],
            320, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(2, 2),
            padding="VALID",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3-6"
        )

        branch7x7x3 = conv(
            net, [1, 1],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7x3_0"
        )
        branch7x7x3 = conv(
            branch7x7x3, [1, 7],
            64, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7x3_1"
        )
        branch7x7x3 = conv(
            branch7x7x3, [7, 1],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7x3_2"
        )
        branch7x7x3 = conv(
            branch7x7x3, [3, 3],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(2, 2),
            padding="VALID",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch7x7x3_3"
        )
        branch_pool = pool(net, (3, 3), stride=(2, 2), padding="VALID", pooling_type="AVG", name="pool11",
                           print_shape=True)
        net = concat(axis=3, values=[branch3x3, branch7x7x3, branch_pool])
        branch1x1 = conv(
            net, [1, 1],
            160, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x1_1-12"
        )
        branch3x3 = conv(
            net, [1, 1],
            224, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3_1-112"
        )
        branch3x3 = concat(axis=3,values=[conv(
            branch3x3, [1, 3],
            192, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3_1-12"
        ),
            conv(
                branch3x3, [3, 1],
                192, weight_initializer={
                    "name": "xavier"
                },
                bias_initializer={
                    "name": "random_normal",
                    "mean": 0.0,
                    "stddev": 1.0
                },
                strides=(1, 1),
                padding="SAME",
                activation_func={"name": "crelu"},
                dropout_func={"keep_prob": 0.5},
                batch_norm_func={"phase_train": train},
                print_shape=True,
                name="branch3x3db1_11-12"
            )])
        branch3x3db1 = conv(
            net, [3, 1],
            224, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db1_1-12"
        )

        branch3x3db1 = conv(
            branch3x3db1, [3, 3],
            192, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db1_1-2_112"
        )
        branch3x3db1 = concat(axis=3,values=[conv(
            branch3x3db1, [1, 3],
            192, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db11_1-12"
        ),
            conv(
                branch3x3db1, [3, 1],
                192, weight_initializer={
                    "name": "xavier"
                },
                bias_initializer={
                    "name": "random_normal",
                    "mean": 0.0,
                    "stddev": 1.0
                },
                strides=(1, 1),
                padding="SAME",
                activation_func={"name": "crelu"},
                dropout_func={"keep_prob": 0.5},
                batch_norm_func={"phase_train": train},
                print_shape=True,
                name="branch3x3db1_111-12"
            )])
        branch_pool = pool(net, (3, 3), stride=(1, 1), padding="SAME", pooling_type="AVG", name="pool111",
                           print_shape=True)
        branch_pool = conv(
            branch_pool, [1, 1],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch_pool_21"
        )
        net = concat(axis=3, values=[branch1x1, branch3x3, branch3x3db1, branch_pool], name="concat9")
        branch1x1 = conv(
            net, [1, 1],
            160, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch1x1_1-13"
        )
        branch3x3 = conv(
            net, [1, 1],
            224, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3_1-113"
        )
        branch3x3 = concat(axis=3, values=[conv(
            branch3x3, [1, 3],
            192, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3_1-13"
        ),
            conv(
                branch3x3, [3, 1],
                192, weight_initializer={
                    "name": "xavier"
                },
                bias_initializer={
                    "name": "random_normal",
                    "mean": 0.0,
                    "stddev": 1.0
                },
                strides=(1, 1),
                padding="SAME",
                activation_func={"name": "crelu"},
                dropout_func={"keep_prob": 0.5},
                batch_norm_func={"phase_train": train},
                print_shape=True,
                name="branch3x3db1_11-13"
            )])
        branch3x3db1 = conv(
            net, [3, 1],
            224, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db1_1-13"
        )

        branch3x3db1 = conv(
            branch3x3db1, [3, 3],
            192, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db1_1-2_113"
        )
        branch3x3db1 = concat(axis=3, values=[conv(
            branch3x3db1, [1, 3],
            192, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch3x3db11_1-13"
        ),
            conv(
                branch3x3db1, [3, 1],
                192, weight_initializer={
                    "name": "xavier"
                },
                bias_initializer={
                    "name": "random_normal",
                    "mean": 0.0,
                    "stddev": 1.0
                },
                strides=(1, 1),
                padding="SAME",
                activation_func={"name": "crelu"},
                dropout_func={"keep_prob": 0.5},
                batch_norm_func={"phase_train": train},
                print_shape=True,
                name="branch3x3db1_111-13"
            )])
        branch_pool = pool(net, (3, 3), stride=(1, 1), padding="SAME", pooling_type="AVG", name="pool_12",
                           print_shape=True)
        branch_pool = conv(
            branch_pool, [1, 1],
            96, weight_initializer={
                "name": "xavier"
            },
            bias_initializer={
                "name": "random_normal",
                "mean": 0.0,
                "stddev": 1.0
            },
            strides=(1, 1),
            padding="SAME",
            activation_func={"name": "crelu"},
            dropout_func={"keep_prob": 0.5},
            batch_norm_func={"phase_train": train},
            print_shape=True,
            name="branch_pool_22"
        )
        net = concat(axis=3, values=[branch1x1, branch3x3, branch3x3db1, branch_pool], name="concat10")
        shape = net.get_shape()
        net = pool(net, shape[1:3], stride=(1, 1), padding="VALID", pooling_type="AVG", name="pool_12",
                           print_shape=True)
        net = dropout(net,0.5,"fc_dropout")
        net = flatten(net)
        logits = fc(aux_logits, n_classes, activation_func=None, weight_initializer={
            "name": "random_normal",
            "mean": 0.0,
            "stddev": 0.001,
            "restore": True
        },
                        bias_initializer={
                            "name": "random_normal",
                            "mean": 0.0,
                            "stddev": 1.0,
                            "restore": True
                        },name="fc_final")

        return logits
