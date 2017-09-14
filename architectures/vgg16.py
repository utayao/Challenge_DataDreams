from nn.layers import *

def get_module(inp,num_classes):
    conv1 = conv_layer(inp,[3,3],64,1,activation="elu",
                        w_initialization="he_normal",b_initialization=0.0,padding="SAME",name="conv1_1")
    conv1 = conv_layer(conv1,[3,3],64,1,activation="elu",
                        w_initialization="he_normal",b_initialization=0.0,padding="SAME",name="conv1_2")
    pool1 = max_pool(conv1,[2,2],[2,2],name="max_pool1")
    conv2 = conv_layer(pool1,[3,3],128,1,activation="elu",
                         w_initialization="he_normal",b_initialization=0.0,padding="SAME",name="conv2_1")
    conv2 = conv_layer(conv2,[3,3],128,1,activation="elu",
                     w_initialization="he_normal",b_initialization=0.0,padding="SAME",name="conv2_2")
    pool2 = max_pool(conv2,[2,2],[2,2],name="max_pool2")

    conv3 = conv_layer(pool2,[3,3],256,1,activation="elu",
                     w_initialization="he_normal",b_initialization=0.0,padding="SAME",name="conv3_1")
    conv3 = conv_layer(conv3,[3,3],256,1,activation="elu",
                     w_initialization="he_normal",b_initialization=0.0,padding="SAME",name="conv3_2")
    conv3 = conv_layer(conv3,[3,3],256,1,activation="elu",
                     w_initialization="he_normal",b_initialization=0.0,padding="SAME",name="conv3_3")
    pool3 = max_pool(conv3,[2,2],[2,2],name="max_pool3")

    conv4 = conv_layer(pool3,[3,3],512,1,activation="elu",
                     w_initialization="he_normal",b_initialization=0.0,padding="SAME",name="conv4_1")
    conv4 = conv_layer(conv4,[3,3],512,1,activation="elu",
                     w_initialization="he_normal",b_initialization=0.0,padding="SAME",name="conv4_2")
    conv4 = conv_layer(conv4,[3,3],512,1,activation="elu",
                     w_initialization="he_normal",b_initialization=0.0,padding="SAME",name="conv4_3")
    pool4 = max_pool(conv4,[2,2],[2,2],name="max_pool4")

    conv5 = conv_layer(pool4,[3,3],512,1,activation="elu",
                     w_initialization="he_normal",b_initialization=0.0,padding="SAME",name="conv5_1")
    conv5 = conv_layer(conv5,[3,3],512,1,activation="elu",
                     w_initialization="he_normal",b_initialization=0.0,padding="SAME",name="conv5_2")
    conv5 = conv_layer(conv5,[3,3],512,1,activation="elu",
                     w_initialization="he_normal",b_initialization=0.0,padding="SAME",name="conv5_3")
    pool5 = max_pool(conv5,[2,2],[2,2],name="max_pool5")
    x = flatten(pool5)
    fc6 = fc(x,4096,activation="elu",w_initialization="he_normal",b_initialization=0.1,name="fc6")
    drop1 = dropout(fc6,0.5,name="dropout_fc6")
    fc7 = fc(drop1,4096,activation="elu",w_initialization="he_normal",b_initialization=0.1,name="fc7")
    drop2 = dropout(fc7,0.5,name="dropout_fc7")
    fc8 = fc(drop2,num_classes,activation=None,w_initialization="he_normal",b_initialization=0.1,name="fc8")
    return fc8
