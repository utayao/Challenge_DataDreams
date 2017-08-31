import tensorflow as tf
import numpy as np
import re
import tf_args
from architectures.slim import slim


class Model(object):
    def __init__(self, net, image_size, n_classes):
        self._net = net
        self._image_size = image_size
        self._n_class = n_classes

    def build_infer_op(self, train, restore_logits):
        with tf.name_scope("inference"):
            self.acc_op = tf.nn.softmax(self.logits)
        return self.acc_op
        batch_norm_params = {
            'decay': tf_args.BATCHNORM_MOVING_AVERAGE_DECAY,
            'epsilon': 0.001,
        }
        with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
            with slim.arg_scope([slim.ops.conv2d],
                                stddev=0.1,
                                activation=tf.nn.relu,
                                batch_norm_params=batch_norm_params):
                logits, end_points = slim.inception.inception_v3(
                    self.data_op,
                    dropout_keep_prob=0.8,
                    num_classes=self._n_class,
                    is_training=train,
                    restore_logits=restore_logits,
                    scope=scope
                )
        self.build_summary_op(end_points)
        auxiliary_logits = end_points['aux_logits']
        return logits, auxiliary_logits, end_points['predictions']

    def build_loss_op(self, logits_op, label_op, batch_size):

        sparse_labels = tf.reshape(label_op, [-1, 1])
        indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
        concated = tf.concat([indices, sparse_labels], 1)
        num_classes = logits_op[0].get_shape()[-1].value
        dense_labels = tf.sparse_to_dense(concated,
                                          [batch_size, num_classes],
                                          1.0, 0.0)
        slim.losses.cross_entropy_loss(logits_op[0],
                                       dense_labels,
                                       label_smoothing=0.1,
                                       weight=1.0)
        slim.losses.cross_entropy_loss(logits_op[1],
                                       dense_labels,
                                       label_smoothing=0.1,
                                       weight=0.4,
                                       scope='aux_loss')

        losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(losses + regularization_losses, name="total_loss")
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg")
        loss_averages_op = loss_averages.apply(losses + [total_loss])
        for l in losses + [total_loss]:
            loss_name = re.sub('%s_[0-9]*/'%tf_args.TOWER_NAME,'',l.op.name)
            tf.summary.scalar(loss_name+' (Raw)', l)
            tf.summary.scalar(loss_name,loss_averages.average(l))
        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)
        return total_loss

    def build_input_op(self, train=False):

        with tf.name_scope("input"):
            self.data_op = tf.placeholder(
                tf.float32,
                shape=[None, self._image_size[0], self._image_size[1], 3]
            )
            self.label_op = tf.placeholder(
                tf.int64,
                shape=[None, self._n_class]
            )
            self.phase_train = tf.placeholder(tf.bool, name="phase_train")

    def build_evaluation_op(self, prob_op, label_op):
        with tf.name_scope("evaluation"):
            recall = tf.metrics.recall(label_op, prob_op)
            precision = tf.metrics.precision(label_op, prob_op)
            f1 = tf.multiply(2.0, tf.divide(tf.multiply(recall, precision), tf.add(recall, precision)))
            accuracy = tf.metrics.accuracy(label_op, prob_op)

        return accuracy, precision, recall, f1

    def build(self, train=True, batch_size=32):
        self.build_input_op(train)

        self.logits, auxiliary_logits, endpoints = self.build_infer_op(train, True)

        self.acc, self.precision, self.recall, self.f1 = self.build_evaluation_op(endpoints['predictions'], self.label_op)
        if train:
            self.build_loss_op(self.logits, self.label_op, batch_size)
            self.loss = self.tower_loss(self.data_op,self.label_op,self._n_class,scope,reuse_variables=None)
            self.summary = self.build_summary_op(endpoints)
    def tower_loss(self,images,labels,n_class,scope,reuse_variables=True):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
            logits =
    @staticmethod
    def _activation_summary(x):
        tensor_name = re.sub('%s_[0-9]*/' % tf_args.TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def build_summary_op(self, end_points):

        with tf.name_scope("summary"):
            for act in end_points.values():
                self._activation_summary(act)
            tf.summary.scalar("loss", self.loss_op)
            acc_op, precision, recall, f1 = self.evaluation_ops
            tf.summary.scalar("accuracy", acc_op[0])
            tf.summary.scalar("precision", precision[0])
            tf.summary.scalar("recall", recall[0])
            tf.summary.scalar("f1", f1[0])

            merged = tf.summary.merge_all()

        return merged

    @property
    def input_ops(self):
        return (
            self.data_op,
            self.label_op,
            self.phase_train
        )

    @property
    def evaluation_ops(self):
        return (
            self.acc,
            self.precision,
            self.recall,
            self.f1
        )

    @property
    def loss_op(self):
        return self.loss

    @property
    def summary_op(self):
        return self.summary

    @property
    def infer_op(self):
        return self.acc_op
