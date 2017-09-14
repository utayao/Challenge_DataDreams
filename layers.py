import math
from cv2 import getGaussianKernel
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
parameters=[]
VARIABLES_TO_RESTORE = '_variables_to_restore_'
def weight_bias(shape, stddev, bias_init=0.1):
    W = tf.Variable(tf.truncated_normal(shape, stddev=stddev), name='weight')
    b = tf.Variable(tf.constant(bias_init, shape=shape[-1:]), name='bias')
    return W, b


def _to_tensor(x,dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype!= dtype:
        x = tf.cast(x,dtype)
    return x
def leaky_relu(alpha,x):
    return tf.maximum(alpha*x,x)

def _two_element_tuple(int_or_list):
    if isinstance(int_or_list,(list,tuple)):
        return int(int_or_list[0]),int(int_or_list[1])
    if isinstance(int_or_list,int):
        return int(int_or_list),int(int_or_list)
    if isinstance(int_or_list,tf.Tensorshape):
        return int_or_list[0],int_or_list[1]
    raise ValueError("Must be int or list ot tensor type")

def get_shape(incoming):
    if isinstance(incoming,tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array,list,tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid layer")
def autoformat_kernel(strides):
    if isinstance(strides,int):
        return [1,strides,strides,1]
    elif isinstance(strides,(tuple,list)):
        if len(strides)==2:
            return [1,strides[0],strides[1],1]
        elif len(strides)==4:
            return [strides[0],strides[1],strides[2],strides[3]]
        else:
            raise Exception("strides error")
    else:
        raise Exception("strides format error")

def get_deconv_filter(f_shape):
    width = f_shape[0]
    height = f_shape[0]
    f = math.ceil(width/2.0)
    c = (2*f-1-f%2)/(2.0*f)
    bilinear = np.zeros([f_shape[0],f_shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1-abs(x/f-c))*(1-abs(y/f-c))
            bilinear[x,y] = value 
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:,:,i,i] = bilinear
    init = tf.constant_initializer(value=weights,dtype=tf.float32)
    return tf.get_variable(name="up_filter",initializer=init,shape=weights.shape)

class Dropout:
    def __init__(self, keep_prob, name='dropout'):
        self.keep_prob = keep_prob
        self.name = name

    def apply(self, x, index, model):
        with tf.name_scope(self.name):
            keep_prob = tf.select(model.is_training, self.keep_prob, 1.0)
            self.h = tf.nn.dropout(x, keep_prob)

            return self.h

class Dense:
    def __init__(self, fan_out, name='dense'):
        self.fan_out = fan_out
        self.name = name

    def apply(self, x, index, model):
        global parameters
        with tf.name_scope(self.name):
            input_shape = x.get_shape()
            fan_in = input_shape[-1].value
            stddev = math.sqrt(1.0 / fan_in) # he init

            shape = [fan_in, self.fan_out]
            W, b = weight_bias(shape, stddev=stddev, bias_init=0.0)
            parameters+=[W,b]
            self.h = tf.matmul(x, W) + b
            
            return self.h

class Activation:
    def __init__(self, activation, name='activation'):
        self.name = name
        self.activation = activation

    def apply(self, x, index, model):
        with tf.name_scope(self.name):
            self.h = self.activation(x)
            return self.h

class LeakyReLU:
    def __init__(self,alpha=0.3):
        self.alpha = alpha
    def __call__(self,x):
        self.alpha = _to_tensor(self.alpha,x.dtype.base_dtype)
        return tf.maximum(self.alpha*x,x)

class PReLU:
    def __init__(self,weights_init="zeros"):
        self.weights_init=weights_init
    def __call__(self,x):
        w_shape = get_shape(x)[1:]
        i_scope = ""
        if hasattr(x,"scope"):
            if x.scope: i_scope = x.scope 
        with tf.name_scope(i_scope+name) as scope:
            w_init = weights.get(self.weights_init)()
            alphas = tf.get_variable(shape=w_shape,initializer=w_init,restore=True,name=scope+"alphas")
            x = tf.nn.relu(x) + tf.mul(alphas,(x-tf.abs(x)))*0.5
            x.scope = scope 
            x.alphas = alphas
            return x

class MaxPool:
    def __init__(self, ksize, strides, padding='SAME', name='max_pool'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.name = name

    def apply(self, x, index, model):
        with tf.name_scope(self.name):
            if isinstance(self.padding,(int,list)):
                pad_h,pad_w=_two_element_tuple(self.padding)
                x=tf.pad(x,[[0,0],[pad_h,pad_h],[pad_w,pad_w],[0,0]],"CONSTANT")
                self.padding="VALID"

            self.h = tf.nn.max_pool(x, self.ksize, self.strides, self.padding)
            return self.h

class GlobalAvgPool:
    def __init__(self, name='global_avg_pool'):
        self.name = name

    def apply(self, x, index, model):
        input_shape = x.get_shape().as_list()
        k_w, k_h = input_shape[1], input_shape[2]
        with tf.name_scope(self.name):
            self.h = tf.nn.avg_pool(x, [1, k_w, k_h, 1], [1, 1, 1, 1], 'VALID')
            return self.h

class DeConv2D:
    def __init__(self,num_classes,shape=None,kernel_size=4,
                    strides=2,trainable=True,restore=True,reuse=False,scope=None,name="deconv2d"):
        
        self.shape = shape
        self.kernel_size = kernel_size
        self.strides = strides
        self.trainable = trainable 
        self.restore = restore 
        self.reuse = reuse 
        self.scope = scope
        self.name = name 
        self.num_classes = num_classes
    def apply(self,x,index,model):
        input_shape = get_shape(x)
        strides = autoformat_kernel(self.strides)
        f_shape = [self.kernel_size,self.kernel_size,self.num_classes,input_shape[-1]]
        with tf.variable_scope([x],self.scope,self.name,reuse=self.reuse) as scope:
            name = scope.name

            if self.shape is None:
                in_shape = tf.shape(x)
                h = ((in_shape[1]-1)*stride) + 1
                w = ((in_shape[2]-1)* stride) + 1
                new_shape = [in_shape[0],h,w,self.num_classes]
            else:
                new_shape= [self.shape[0],self.shape[1],self.shape[2],self.num_classes]
            output_shape = tf.pack(new_shape) 
            weights = get_deconv_filter(f_shape)
        
            deconv = tf.nn.conv2d_transpose(x,weights,output_shape,strides=strides,padding="SAME")
            deconv.scope = scope 
        return deconv

class blur:
    def __init__(self,size):
        self.size = size 
    def apply(self,x,index,model):
        gaussian = getGaussianKernel(self.size)
        pad = math.floor(size/2)
        padding = tf.pad(x,[[0,0],[pad,pad],[pad,pad],[0,0]],"CONSTANT")
        gaussian = gaussian/np.sum(gaussian)*0.5 
        blur_image = tf.nn.conv2d(padding,gaussian,strides=[1,1,1,1],padding="VALID")
        
        return blur_image



class AvgPool:
    def __init__(self, ksize, strides, padding='VALID', name='avg_pool'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.name = name

    def apply(self, x, index, model):
        with tf.name_scope(self.name):
            if isinstance(self.padding,(int,list)):
                pad_h,pad_w=_two_element_tuple(self.padding)
                x=tf.pad(x,[[0,0],[pad_h,pad_h],[pad_w,pad_w],[0,0]],"CONSTANT")
                self.padding="VALID"
            self.h = tf.nn.avg_pool(x, self.ksize, self.strides, self.padding)
            return self.h

class Input:
    def __init__(self, input_placeholder):
        self.h = input_placeholder

    def apply(self, x, index, model):
        return self.h


class Conv2D:
    def __init__(self, filter_shape, output_channels, strides, padding='SAME', name='conv2d'):
        self.filter_shape = filter_shape
        self.output_channels = output_channels
        self.strides = strides
        self.padding = padding
        self.name = name

    def apply(self, x, index, model):
        global parameters
        with tf.name_scope(self.name):
            if isinstance(self.padding,(int,list)):
                pad_h,pad_w=_two_element_tuple(self.padding)
                x=tf.pad(x,[[0,0],[pad_h,pad_h],[pad_w,pad_w],[0,0]],"CONSTANT")
                self.padding="VALID"
            input_shape = x.get_shape()
            input_channels = input_shape[-1].value

            k_w, k_h = self.filter_shape
            stddev = math.sqrt(2.0 / ((k_w * k_h) * input_channels)) # he init

            shape = self.filter_shape + [input_channels, self.output_channels]
            W, b = weight_bias(shape, stddev=stddev, bias_init=0.0)
            parameters+=[W,b]
            self.h = tf.nn.conv2d(x, W, self.strides, self.padding) + b
            
            return self.h

class LRN:
    def __init__(self,radius,alpha,beta,bias,name="lrn"):
        self.radius = radius
        self.alpha = alpha
        self.beta = beta 
        self.bias = bias 
        self.name = name 
    def apply(self,x,index,model):
        lrn = tf.nn.local_response_normalization(x,
                                                depth_radius=self.radius,
                                                alpha=self.alpha,
                                                beta=self.beta,
                                                bias=self.bias)
        return lrn
class Flatten:
    def __init__(self, name='flatten'):
        self.name = name

    def apply(self, x, index, model):
        with tf.name_scope(self.name):
            shape = x.get_shape()
            dim = shape[1] * shape[2] * shape[3]
            self.h = tf.reshape(x, [-1, dim.value])
            return self.h

class ConvFactory:
    def __init__(self,kernel_size,fan_out,strides=[1,1,1,1],padding="SAME",act_type=tf.nn.relu,name="confactory"):
        
        self.conv=Conv2D(kernel_size,fan_out,strides,padding=padding,name="conv_%s"%name)
        self.batchnorm=Conv2DBatchNorm(fan_out)
        self.Activation=Activation(act_type)
        self.act_type=act_type
    def apply(self,x,index,model):
        prev=self.conv.apply(x,index,model)
        prev=self.batchnorm.apply(prev,index,model)
        prev=self.Activation.apply(prev,index,model)
        return prev

class InceptionFactoryA:
    def __init__(self,num_1x1,num_3x3red,num_3x3,num_d3x3red,num_d3x3,pool,proj,name):
        self.c1x1=ConvFactory([1,1],num_1x1,[1,1,1,1],name="%s_1x1"%name)
        self.c3x3r=ConvFactory([1,1],num_3x3red,name="%s_3x3r"%name)
        self.c3x3=ConvFactory([3,3],num_3x3,padding=[1,1],name="%s_3x3"%name)
        self.cd3x3r=ConvFactory([1,1],num_d3x3red,name="%s_d_3x3"%name)
        self.cd3x3=ConvFactory([3,3],num_d3x3,padding=[1,1],name="%s_d_3x3_0"%name)
        self.cd3x3_1=ConvFactory([3,3],num_d3x3,padding=[1,1],name="%s_d_3x3_1"%name)
        if pool=="max":

            self.pool=MaxPool(ksize=[1,3,3,1],strides=[1,1,1,1],padding=[1,1],name="%s_pool"%name)
        elif pool=="avg":
            self.pool=AvgPool(ksize=[1,3,3,1],strides=[1,1,1,1],padding=[1,1],name="%s_pool"%name)
        self.cproj=ConvFactory([1,1],proj,name="%s_proj"%name)
    def apply(self,x,i,model):
        prev_1=self.c1x1.apply(x,i,model)
        prev_2=self.c3x3r.apply(x,i,model)
        prev_3=self.c3x3.apply(prev_2,i,model)
        prev_4=self.cd3x3r.apply(x,i,model)
        prev_5=self.cd3x3.apply(prev_4,i,model)
        prev_6=self.cd3x3_1.apply(prev_5,i,model)
        prev_7=self.pool.apply(x,i,model)
        prev_8=self.cproj.apply(prev_7,i,model)
        concat=tf.concat(3,[prev_1,prev_3,prev_6,prev_8])
        return concat

class InceptionFactoryB:
    def __init__(self,num_3x3red,num_3x3,num_d3x3red,num_d3x3,name):
        self.c3x3r=ConvFactory([1,1],num_3x3red,name="%s_3x3"%name)
        self.c3x3=ConvFactory([3,3],num_3x3,padding=[1,1],strides=[1,2,2,1],name="%s_3x3_0"%name)
        self.cd3x3r=ConvFactory([1,1],num_d3x3red,name="%s_d_3x3"%name)
        self.cd3x3=ConvFactory([3,3],num_d3x3,padding=[1,1],strides=[1,1,1,1],name="%s_d_3x3_1"%name)
        self.cd3x3_1=ConvFactory([3,3],num_d3x3,padding=[1,1],strides=[1,2,2,1],name="%s_d_3x3_2"%name)
        self.pool=MaxPool(ksize=[1,3,3,1],strides=[1,2,2,1],padding=[1,1],name="pool_%s"%name)

    def apply(self,x,i,model):
        c3x3r=self.c3x3r.apply(x,i,model)
        c3x3=self.c3x3.apply(c3x3r,i,model)
        cd3x3r=self.cd3x3r.apply(x,i,model)
        cd3x3=self.cd3x3.apply(cd3x3r,i,model)
        cd3x3=self.cd3x3_1.apply(cd3x3,i,model)
        #prev_6=self.cd3x3_1.apply(prev_5,i,model)
        prev_7=self.pool.apply(x,i,model)
        concat=tf.concat(3,[c3x3,cd3x3,prev_7])
        return concat



class Conv2DBatchNorm:
    
    def __init__(self, fan_out, affine=True, name='batch_norm'):
        self.fan_out = fan_out
        self.affine = affine
        self.name = name

    def apply(self, x, index, model):
        with tf.name_scope(self.name):
            beta = tf.Variable(tf.constant(0.0, shape=[self.fan_out]), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[self.fan_out]), name='gamma', trainable=self.affine)

            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')

            ema = tf.train.ExponentialMovingAverage(decay=0.9)
            ema_apply_op = ema.apply([batch_mean, batch_var])
            ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

            def mean_var_with_update():
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = control_flow_ops.cond(model.is_training, mean_var_with_update, lambda: (ema_mean, ema_var))

            self.h = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, self.affine)
            return self.h
