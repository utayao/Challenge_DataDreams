import tensorflow as tf 
#import train
from nn.metrics import hamming_score
import nn.layers 
import os
from architectures import Inception_BN_Modified as mod
#from architectures import vgg16 as modd
#import nn.image_processing as image_processing
from image_processing import global_funcs
import utils
from layers import Input
import pdb
import sys 
import numpy as np


flags = tf.app.flags
FLAGS = flags.FLAGS 

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

class Model:
    def __init__(self,fold,data_augment_params):
        self.fold = fold 
        self.data_params = data_augment_params
        self.sess = tf.get_default_session()
        
        self.checkpoint = makedirs(os.path.join(FLAGS.checkpoint_path,"model_%s"%self.fold))
        self.summaries = makedirs(os.path.join(FLAGS.summary_path,"model_%s"%self.fold))
        self.models = makedirs(os.path.join(FLAGS.model_path,"mdoel_%s"%self.fold))
        self.x = tf.placeholder(tf.float32,shape=[None,224,224,3])
        self.y_ = tf.placeholder(tf.float32,shape=[None,FLAGS.num_classes])
        #self.y = mod.get_module(self.x,FLAGS.num_classes)
        global_step = tf.Variable(0.0)
        self.global_step_op = global_step.assign_add(1)
        self.is_training = tf.placeholder(tf.bool,shape = [])
        #layers =mod.Inception_BN(FLAGS.num_classes)
        #prev = None
        #layers = [Input(self.x)] + layers 
        #for i,layer in enumerate(layers):
            #prev = layer.apply(prev,i,self)
        
        #self.y = prev 
        self.y = mod.get_module(self.x,FLAGS.num_classes)
        #print nn.layers.get_parameters
        #print ly.parameters
        if FLAGS.pre_trained:
            weights = np.load(FLAGS.weight_file)
            keys = sorted(weights.keys())
            #print keys

            for i ,k in enumerate(keys):
                #print weights[k]
                #print weights[k].shape
                #return
                if "fc8" not in k:
                    #print weights[k]
                    #print weights[k].shape
                    self.sess.run(nn.layers.get_parameters[i].assign(weights[k]))
                    #for j,v in weights[k].items():
                        #if len(weights[k][j].shape!=len(ly.parameters[index].get_shape().as_list())):
                            #self.sess.run(ly.parameters[index].assign(weights[k][j]))
                        #index+=1

            print "loaded pre trained weights"
        with tf.name_scope("loss"):
            self.loss_op = tf.nn.softmax_cross_entropy_with_logits(logits=self.y,labels=self.y_)
            self.loss_op = tf.reduce_mean(self.loss_op,name="loss")
            tf.summary.scalar("loss",self.loss_op)
        with tf.name_scope("accuracy"):
            self.correct_prediction = tf.equal(tf.argmax(self.y,1),tf.argmax(self.y_,1))
            
            self.correct_prediction = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))
            tf.summary.scalar("accuracy",self.correct_prediction)

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(self.loss_op,name="train")
        
        self.summaries_op = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)
        self.summary_writer = tf.summary.FileWriter(FLAGS.summary_path,self.sess.graph_def)
        #print self.models
        tf.train.write_graph(self.sess.graph_def,self.models,"model.pb",as_text=False)
        self.ckpt = tf.train.get_checkpoint_state(self.checkpoint)

        if self.ckpt and self.ckpt.model_checkpoint_path:
            print "ckpt loading from {}".format(self.checkpoint)
            self.saver.restore(self.sess,self.ckpt.model_checkpoint_path)
        else:
            print "ckpt not found"
        
    def data_augment(self,data_augment,batchs):
        for i in range(batchs.shape[0]):
            batch_x = batchs[i, ...]
            utils.save_image(batch_x,'sample_0.png')
            for func,params in data_augment.items():
                if np.random.uniform() > 0.5:
                    batch_x = global_funcs(func)(batch_x, params)
            utils.save_image(batch_x,'sample_1.png')
            batchs[i,...] = batch_x
            break
        return batchs



     
    def fit(self,epoch,x_train,y_train,is_training=True):
        
        length = x_train.shape[0]
        total_loss = []
        total_acc = []
        summary = None 
        for start in range(0,length,FLAGS.batch_size):
            end = min(FLAGS.batch_size+start,length)
            print "running {}/{}\r".format(start,end),
            sys.stdout.flush()
            batch_x = x_train[start:end] 
            batch_y = y_train[start:end] 
            batch_x = batch_x.astype(np.float32)
            batch_y  = batch_y.astype(np.float32)
            #batch_x = np.swapaxes(batch_x,1,2)
            #print batch_y.max()
            #mean = np.mean(batch_x)
            #std = np.std(batch_x)
            #batch_x = (batch_x-mean)/std
            #batch_x/=255.0
            if is_training:
                #pdb.set_trace()
                batch_x = self.data_augment(self.data_params,batchs=batch_x)
            batch_x/=255.0
            #print batch_x.dtype
            #batch_y = batch_y.astype(np.float32)
            #print batch_y.max()
            #batch_y = batch_y.reshape((-1,224,224,1))
            #mean_pixel=[103.939,116.79,123.68]
            #for i in range(3):
                #batch_x[:,:,:,i] = batch_x[:,:,:,i]-mean_pixel[i]
            #batch_x/=255.0
            
            #print x_train.shape,y_train.shape    
            _,loss,acc,summary = self.sess.run([self.train_op,self.loss_op,self.correct_prediction,self.summaries_op],feed_dict={
                self.x:batch_x,
                self.y_:batch_y,
                self.is_training:is_training
                })
                
            total_loss.append(loss)
            total_acc.append(acc)

        if is_training:
            self.saver.save(self.sess,self.checkpoint+"/model.ckpt",global_step=epoch)
            self.summary_writer.add_summary(summary,global_step=epoch)

        return np.mean(total_loss),np.mean(total_acc)

    

        


