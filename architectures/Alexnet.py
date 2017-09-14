from  nn.layers import *
import pdb


def get_module(inp,num_classes):
    #pdb.set_trace()
    #inp = input_data(inp_shape)
    conv1 = conv_layer(inp,[11,11],96,4,
                        w_initialization="xavier",b_initialization=0.0,
                        batch_norm=False,activation="relu",name="conv1",pre_trained=FLAGS.pre_trained
                        )
    print conv1.get_shape()
    lrn1 = lrn(conv1,radius=2,alpha=2e-05,beta=0.75,bias=1.0)
    pool1 = max_pool(lrn1,[3,3],[2,2],padding="VALID")
    print pool1.get_shape()
    conv2 = conv_layer(pool1,[5,5],512,1,
                     w_initialization="xavier",b_initialization=0.0,
                     batch_norm=False,activation="relu",name="conv2",pre_trained=FLAGS.pre_trained
                     )
    print conv2.get_shape()
    lrn2 = lrn(conv2,radius=2,alpha=2e-05,beta=0.75,bias=1.0)
    pool2 = max_pool(lrn2,[3,3],[2,2],padding="VALID")
    print pool2.get_shape()
    conv3 = conv_layer(pool2,[3,3],384,1,
                  w_initialization="xavier",b_initialization=0.0,
                  batch_norm=False,activation="relu",name="conv3",pre_trained=FLAGS.pre_trained
                  )
    print conv3.get_shape()
    #lrn3 = lrn(conv3,radius=2,alpha=2e-05,beta=0.75,bias=1.0)
    #pool3 = max_pool(lrn3,[3,3],[2,2],padding="VALID")
    #print pool3.get_shape()
    conv4 = conv_layer(conv3,[3,3],768,1,
               w_initialization="xavier",b_initialization=0.0,
               batch_norm=False,activation="relu",name="conv4",pre_trained=FLAGS.pre_trained
               )
    print conv4.get_shape()
    #lrn4 = lrn(conv4,radius=2,alpha=2e-05,beta=0.75,bias=1.0)
    pool4 = max_pool(conv4,[3,3],[2,2],padding="VALID")
    print pool4.get_shape()
    conv5 = conv_layer(pool4,[3,3],512,1,
            w_initialization="xavier",b_initialization=0.0,
            batch_norm=False,activation="relu",name="conv5",pre_trained=FLAGS.pre_trained
            )
    print conv5.get_shape()
    #lrn5 = lrn(conv5,radius=2,alpha=2e-05,beta=0.75,bias=1.0)
    #pool5 = max_pool(lrn5,[3,3],[2,2],padding="VALID")
    x = flatten(conv5)
    fc6 = fc(x,4096,activation="relu",w_initialization="xavier",b_initialization=0.1,name="fc6",pre_trained=FLAGS.pre_trained)
    drop1 = dropout(fc6,0.5,name="dropout_fc6")
    fc7 = fc(drop1,4096,activation="relu",w_initialization="xavier",b_initialization=0.1,name="fc7",pre_trained=FLAGS.pre_trained)
    drop2 = dropout(fc7,0.5)
    fc8 = fc(drop2,num_classes,activation=None,w_initialization="xavier",b_initialization=0.1,name="fc8",pre_trained=FLAGS.pre_trained)
    return fc8


