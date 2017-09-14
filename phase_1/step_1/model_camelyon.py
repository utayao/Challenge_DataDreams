import tensorflow as tf
import numpy as np
import re
import tf_args
from architectures.slim import slim


class Model(object):
    def __init__(self, image_size, n_classes):
        self._image_size = image_size
        self._n_class = n_classes

    def build_infer_op(self, train, restore_logits):
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
                    restore_logits=restore_logits
                )
        auxiliary_logits = end_points['aux_logits']
        return logits, auxiliary_logits, end_points, end_points['predictions']

    def build_loss_op(self, logits_op,aux_logits, label_op, batch_size):

        # sparse_labels = tf.reshape(label_op, [-1, 1])
        # indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
        # concated = tf.concat([indices, sparse_labels], 1)
        # num_classes = logits_op.get_shape()[-1].value
        # dense_labels = tf.sparse_to_dense(concated,
        #                                   [batch_size, num_classes],
        #                                   1.0, 0.0)
        slim.losses.cross_entropy_loss(logits_op,
                                       label_op,
                                       label_smoothing=0.1,
                                       weight=1.0)
        slim.losses.cross_entropy_loss(aux_logits,
                                       label_op,
                                       label_smoothing=0.1,
                                       weight=0.4,
                                       scope='aux_loss')

        losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(losses + regularization_losses, name="total_loss")
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg")
        loss_averages_op = loss_averages.apply(losses + [total_loss])
        for l in losses + [total_loss]:
            loss_name = re.sub('%s_[0-9]*/' % tf_args.TOWER_NAME, '', l.op.name)
            tf.summary.scalar(loss_name + ' (Raw)', l)
            tf.summary.scalar(loss_name, loss_averages.average(l))
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
                tf.int32,
                shape=[None, self._n_class]
            )
            self.phase_train = tf.placeholder(tf.bool, name="phase_train")

    def build_evaluation_op(self, prob_op, label_op):
        print prob_op.get_shape(),label_op.get_shape()
        predictions = tf.argmax(prob_op,1)
        actuals = tf.argmax(label_op,1)
        #ones_like_actuals = tf.ones(actuals)
        #zeros_like_actuals = tf.zeros_like(actuals)
        #ones_like_predictions = tf.ones_like(predictions)
        #zeros_like_predictions = tf.zeros_like(predictions)

        with tf.name_scope("evaluation"):
#            tp_op = tf.reduce_sum(
#                    tf.cast(
#                        tf.logical_and(
#                            tf.equal(actuals,ones_like_actuals),
#                            tf.equal(predictions,ones_like_predictions)
#                            ),
#                        "float"
#                        )
#                    )
#            tn_op = tf.reduce_sum(
#                    tf.cast(
#                        tf.logical_and(
#                            tf.equal(actuals,zeros_like_actuals),
#                            tf.equal(predictions,zeros_like_predictions)
#                            ),
#                        "float"
#                        )
#                    )
#            fp_op = tf.reduce_sum(
#                    tf.cast(
#                        tf.logical_and(
#                            tf.equal(actuals,zeros_like_actuals),
#                            tf.equal(predictions,ones_like_predictions)
#                            ),
#                        "float"
#                        )
#                    )
#            fn_op = tf.reduce_sum(
#                    tf.cast(
#                        tf.logical_and(
#                            tf.equal(actuals,ones_like_actuals),
#                            tf.equal(predictions,zeros_like_predictions)
#                            ),
#                        "float"
#                        )
#                    )
#            
            tp_op = tf.count_nonzero(predictions * actuals)
            tn_op = tf.count_nonzero((predictions-1) * (actuals -1))
            fn_op = tf.count_nonzero(actuals * (predictions - 1))
            fp_op = tf.count_nonzero(predictions * (actuals - 1))

            recall = tf.divide(tp_op,tf.add(tp_op,fn_op))
            precision = tf.divide(tp_op,tf.add(tp_op,fp_op))
           #accuracy = tf.divide(tf.add(tp_op, tn_op), tf.add(tf.add(tn_op,fp_op),tf.add(fn_op,tp_op)))
           #f1 = tf.multiply(2.0, tf.divide(tf.multiply(recall, precision), tf.add(recall, precision)))
            f1 = (2*precision * recall)/ (precision + recall)
            accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions,actuals)))

        return accuracy, precision, recall, f1

    def build(self, train=True, batch_size=32):
        self.build_input_op(train)

        self.logits, auxiliary_logits, endpoints, self.acc_op = self.build_infer_op(train, True)
        self.acc, self.precision, self.recall, self.f1 = self.build_evaluation_op(endpoints['predictions'],
                                                                                 self.label_op)
        self.prob_op = endpoints['predictions']
        self.class_op = tf.argmax(self.prob_op, 1)

        if train:
            self.loss = self.build_loss_op(self.logits, auxiliary_logits, self.label_op, batch_size)
            self.build_summary_op(endpoints)

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
            tf.summary.scalar("accuracy", acc_op)
            tf.summary.scalar("precision", precision)
            tf.summary.scalar("recall", recall)
            tf.summary.scalar("f1", f1)

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

    @property
    def validation_infer_ops(self):
        return (
                self.prob_op,
                self.class_op
                )
