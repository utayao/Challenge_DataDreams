import tensorflow as tf


class Model(object):
    def __init__(self, net, image_size=(224, 224), n_classes=2):
        self._net = net
        self._image_size = image_size
        self._n_class = n_classes

    def build_infer_op(self):
        with tf.name_scope("inference"):
            self.acc_op = tf.nn.softmax(self.logits)

    def build_loss_op(self, logits_op, label_op, class_weights=None):
        with tf.name_scope("xentropy_loss"):
            if not class_weights:
                return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_op, label_op))
            else:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
                weight_map = tf.multiply(label_op, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)
                loss_map = tf.nn.softmax_cross_entropy_with_logits(logits_op, label_op)
                weights_loss = tf.multiply(loss_map, weight_map)
                return tf.reduce_mean(weights_loss)

    def build_input_op(self, train=False):

        with tf.name_scope("input"):
            self.data_op = tf.placeholder(
                tf.float32,
                shape=[None, self._image_size[0], self._image_size[1], 3]
            )
            self.label_op = tf.placeholder(
                tf.float32,
                shape=[None, self._n_class]
            )
            self.phase_train = tf.placeholder(tf.bool, name="phase_train")

    def build_evaluation_op(self, prob_op, label_op):
        with tf.name_scope("evaluation"):
            recall = tf.metrics.recall(label_op, prob_op)
            precision = tf.metrics.precision(label_op, prob_op)
            f1 = tf.multiply(2, tf.divide(tf.multiply(recall, precision), tf.add(recall, precision)))

            accuracy = tf.metrics.accuracy(label_op, prob_op)
        return accuracy, precision, recall, f1

    def build(self, train=True):
        self.build_input_op(train)
        self.logits = self._net(self.data_op, self._n_class, self.phase_train)
        self.prob_op = self.build_infer_op()
        self.acc = self.build_evaluation_op(self.prob_op, self.label_op)
        if train:
            self.loss = self.build_loss_op(self.logits, self.label_op)
            self.summary = self.build_summary_op()

    def build_summary_op(self):

        with tf.name_scope("summary"):
            tf.summary.scalar("loss", self.loss_op)
            acc_op, precision, recall, f1 = self.evaluation_op
            tf.summary.scalar("accuracy", acc_op)
            tf.summary.scalar("precision", precision)
            tf.summary.scalar("recall", recall)
            tf.summary.scalar("f1", f1)

            merged = tf.summary.merge.all()

        return merged
    