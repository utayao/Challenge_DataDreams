import sys
from step_1 import tf_args

sys.path.append("../")
import math
import pdb
import tensorflow as tf
from utils.data_generator import TrainDataGenerator
import numpy as np
from model_camelyon import Model

FLAGS = tf.app.flags.FLAGS

from utils.data import JobDir
from time import clock
from utils.utils import log_to_file,get_state,save_json

class NetTrainer(object):
    def __init__(self,  data_dir, train_dir, cancer_data_augmentation=None, non_cancer_data_augmentation=None,
                 cv=2, subset=True, image_resize=(572, 572), n_classes=2,
                 batch_size=32,
                 normalize=False):
        self._net = Model(image_resize, n_classes)
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.cross_validaiton = cv
        self._job_dir = JobDir(self.train_dir)
        self._batch_size = batch_size
        self._train_data_loader = TrainDataGenerator(
            train_dir=data_dir, cancer_data_augmentation=cancer_data_augmentation,
            non_cancer_data_augmentation=non_cancer_data_augmentation,
            shuffle=True,
            cv=cv,
            image_resize=image_resize,
            subset=subset,
            normalize=normalize
        )
        self.global_step = None
        self.saver = None
        self.train_op = None

    def build(self):
        self._net.build(train=True,batch_size=self._batch_size)
        self.global_step = tf.Variable(0, trainable=False)
        self.build_train_op(
            self._net.loss,
            optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate),
            global_step=self.global_step
        )
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def _average_gradients(self, tower_grads):

        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def build_train_op(self, loss_op, global_step=None, optimizer=None):
        """
        Build train operator for trainer
        :param loss_op:
        :param global_step:
        :param optimizer:
        :param max_gradient_norm:
        :return:
        """
        with tf.name_scope("train"):
            gradient_maps = optimizer.compute_gradients(loss_op)
            grads = self._average_gradients([gradient_maps])
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            train_op = optimizer.apply_gradients(
                gradient_maps, global_step=global_step
            )
        self.train_op = train_op
        self.optimizer = optimizer

    def batch_data(self, cv, train):
        """ produce a batch of training data """
        if train:

            return self._train_data_loader.sample_batch(self._batch_size, cv, index=0)
        else:
            return self._train_data_loader.sample_batch(self._batch_size, cv, index=1)

    def feed_dict(self, cv, train=True):
        """
        Feed data to tf
        :param batch_size: batch size
        :param cv: cross validation index
        :return: return dict of input and labels
        """
        data_op, label_op, phase_train = self._net.input_ops
        data, label = self.batch_data(cv, train)
        return {data_op: data, label_op: label, phase_train: train}

    def restore(self, sess, checkpoint_path):
        reader = tf.train.NewCheckpointReader(checkpoint_path)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
        saved_names = set([var for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
        uninitialized_variables = set(tf.global_variables()) - saved_names
        restore_vars = []
        name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:

                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
        saver = tf.train.Saver(restore_vars)
        saver.restore(sess, checkpoint_path)
        sess.run(tf.variables_initializer(uninitialized_variables))

    def train(self, max_iter=1000, sess=None):
        """
        Train the model
        :param max_iter:
        :param sess:
        :return:
        """
        # config = tf.ConfigProto(allow_soft_placement=True)
        config = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        # config.gpu_options.allow_growth = True
        # cross validate
        variables_averages = tf.train.ExponentialMovingAverage(
            tf_args.MOVING_AVERAGE_DECAY, self.global_step
        )
        variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variables_averages.apply(variables_to_average)
        batch_norm_updates = tf.get_collection(tf_args.UPDATE_OPS_COLLECTION)
        batch_norm_updates_op = tf.group(*batch_norm_updates)
        self.train_op = tf.group(self.train_op, variables_averages_op, batch_norm_updates_op)

        for each_cross_validation in range(self.cross_validaiton):

            # Initialize session
            sess = sess or tf.Session(config=tf.ConfigProto(gpu_options=config))
            # sess = sess or tf.Session(config=config)
            # Get train log
            train_writer = tf.summary.FileWriter(
                self._job_dir.join_path(
                    self._job_dir.join_path(self._job_dir.log_dir, "train"),
                    "cv_{}".format(each_cross_validation)
                ), sess.graph
            )
            self.summary_op = tf.summary.merge_all()
            ckpt_path = self._job_dir.join_path(
                self._job_dir.checkpoint_path, "cv_{}".format(each_cross_validation)
            )
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            if ckpt:
                print "checkpoint available"
                self.restore(sess, ckpt.model_checkpoint_path)
                print 'weights restored'
            else:
                sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            print "sssssss"
            # Build Forward ops
            run_ops = ([self.train_op, self._net.loss_op, self.summary_op] +
                       list(self._net.evaluation_ops))
            eval_ops = run_ops[1:]
            prev_train_loss = float(get_state(FLAGS.state_file,"prev_train_loss"))
            #self.feed_dict(each_cross_validation)        
            for iter_idx in range(max_iter):
                start = clock()
#                lr = sess.run(self.optimizer._lr)

                _, train_loss, summary, acc, precision, recall, f1 = sess.run(run_ops,feed_dict=self.feed_dict(each_cross_validation))
                end = clock()
                print "TRAINING:- Time:{} per sec, loss: {}, accuracy: {}, precision: {},Recall: {},f1: {}".format(
                        (end - start), train_loss, acc, precision, recall, f1
                    )

                train_writer.add_summary(summary, self.global_step.eval(session=sess))
                # Display the values
                save_json(FLAGS.state_file,"train_loss",train_loss,Type="list")
                save_json(FLAGS.state_file,"train_acc",acc,Type="list")
 
                if iter_idx and iter_idx % tf_args.DISPLAY_ITERS == 0:
                    end = clock()
                    log_to_file("log.txt","TRAINING:- Time:{} per sec, loss: {}, accuracy: {}, precision: {},Recall: {},f1: {}".format(
                        (end - start), train_loss, acc, precision, recall, f1
                    ))
                if iter_idx and iter_idx % tf_args.EVAL_ITERS == 0:
                    loss_arr = []
                    acc_arr = []
                    precision_arr = []
                    recall_arr = []
                    f1_arr = []
                    for eval_start in range(tf_args.EVAL_COUNT):

                        
                        eval_loss, summary, eval_acc, eval_precision, eval_recall, eval_f1 = sess.run(eval_ops,
                                                                                                         feed_dict=self.feed_dict(
                                                                                                             each_cross_validation,
                                                                                                            train=False))
                        print "losss:{} acc: {} precision: {} f1: {}".format(eval_loss,eval_acc,eval_precision,eval_f1)
                        loss_arr.append(eval_loss)
                        acc_arr.append(eval_acc)
                        precision_arr.append(eval_precision)
                        recall_arr.append(eval_recall)
                        f1_arr.append(eval_f1)
                    #pdb.set_trace()
                    log_to_file("log.txt","EVALUATION:- loss: {}, acc: {}, precision: {}, recall: {}, f1: {} ".format(
                        np.mean(loss_arr), np.mean(acc_arr), np.mean(precision_arr), np.mean(recall_arr),
                        np.mean(f1_arr)
                    ))
                    save_json(FLAGS.state_file,"val_loss",np.mean(loss_arr),Type="list")
                    save_json(FLAGS.state_file,"val_acc",np.mean(acc_arr),Type="list")

                if train_loss < prev_train_loss:
                    prev_train_loss = train_loss
                    save_json(FLAGS.state_file,"prev_train_loss",prev_train_loss)
                    end = clock()
                    print("Saving the checkpoint")
                    log_to_file(
                        "log.txt","SAVING with TRAINING:- Time:{} per sec, loss: {}, accuracy: {}, precision: {},Recall: {},f1: {}".format(
                            (end - start), train_loss, acc, precision, recall, f1
                        ))
                    self.saver.save(sess, self._job_dir.join_path(ckpt_path, "weights-ckpt"),
                                    global_step=self.global_step)
