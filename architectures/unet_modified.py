from nn.layers import *

def get_module(inp,num_classes):
    conv1 = conv_layer(inp,[3,3],32,
                    w_initialization="he_normal",b_initialization=0,padding="SAME",
                    activation="relu",batch_norm=True,regularizer="L2",name="conv1_1")
    conv1 = conv_layer(conv1,[3,3],32,
                        w_initialization="he_normal",b_initialization=0,padding="SAME",
                        activation="relu",batch_norm=True,regularizer="L2",name="conv1_2")
    pool1 = max_pool(conv1,[2,2],[2,2],name="max_pool1",padding="SAME")
    #pool1 = dropout(pool1,0.3)
    conv2 = conv_layer(pool1,[3,3],64,
                        w_initialization="he_normal",b_initialization=0,padding="SAME",
                        activation="relu",batch_norm=True,regularizer="L2",name="conv2_1")
    conv2 = conv_layer(conv2,[3,3],64,
                        w_initialization="he_normal",b_initialization=0,padding="SAME",
                        activation="relu",batch_norm=True,regularizer="L2",name="conv2_2")
    pool2 = max_pool(conv2,[2,2],[2,2],name="max_pool2",padding="SAME")
    #pool2 = dropout(pool2,0.4)
    conv3 = conv_layer(pool2,[3,3],128,
                    w_initialization="he_normal",b_initialization=0,padding="SAME",
                    activation="relu",batch_norm=True,regularizer="L2",name="conv3_1")
    conv3 = conv_layer(conv3,[3,3],128,
                         w_initialization="he_normal",b_initialization=0,padding="SAME",
                         activation="relu",batch_norm=True,regularizer="L2",name="conv3_2")
    pool3 = max_pool(conv3,[2,2],[2,2],name="max_pool3",padding="SAME")
    #pool3 = dropout(pool3,0.5)
    conv4 = conv_layer(pool3,[3,3],256,
                     w_initialization="he_normal",b_initialization=0,padding="SAME",
                     activation="relu",batch_norm=True,regularizer="L2",name="conv4_1")
    conv4 = conv_layer(conv4,[3,3],256,
                         w_initialization="he_normal",b_initialization=0,padding="SAME",
                         activation="relu",batch_norm=True,regularizer="L2",name="conv4_2")
    pool4 = max_pool(conv4,[2,2],[2,2],name="max_pool4",padding="SAME")
    #pool4 = dropout(pool4,0.5)
    conv5 = conv_layer(pool4,[3,3],512,
                     w_initialization="he_normal",b_initialization=0,padding="SAME",
                     activation="relu",batch_norm=True,regularizer="L2",name="conv5_1")
    conv5 = conv_layer(conv5,[3,3],512,
                     w_initialization="he_normal",b_initialization=0,padding="SAME",
                     activation="relu",batch_norm=True,regularizer="L2",name="conv5_2")
    #pool5 = max_pool(conv5,[2,2],[2,2],name="max_pool5",padding="SAME")
    #conv5 = dropout(conv5,0.5)
    up = upsampling(conv5,[2,2])
    up6 = tf.concat(3,[up,conv4])
    conv6 = conv_layer(up6,[3,3],256,
            w_initialization="he_normal",b_initialization=0,padding="SAME",
            activation="relu",batch_norm=True,regularizer="L2",name="conv6_1")
    conv6 =conv_layer(conv6,[3,3],256,
            w_initialization="he_normal",b_initialization=0,padding="SAME",
            activation="relu",batch_norm=True,regularizer="L2",name="conv6_2")
    #conv6 = dropout(conv6,0.5)
    up7 = upsampling(conv6,[2,2])
    up7 = tf.concat(3,[up7,conv3])
    conv7 = conv_layer(up7,[3,3],128,
            w_initialization="he_normal",b_initialization=0,padding="SAME",
            activation="relu",batch_norm=True,regularizer="L2",name="conv7_1")
    conv7 = conv_layer(conv7,[3,3],128,
            w_initialization="he_normal",b_initialization=0,padding="SAME",
            activation="relu",batch_norm=True,regularizer="L2",name="conv7_2")
    #conv7 = dropout(conv7,0.4)
    up8 = upsampling(conv7,[2,2])
    up8 = tf.concat(3,[up8,conv2])
    conv8 = conv_layer(up8,[3,3],64,
                         w_initialization="he_normal",b_initialization=0,padding="SAME",
                         activation="relu",batch_norm=True,regularizer="L2",name="conv8_1")
    conv8 = conv_layer(conv8,[3,3],64,
                     w_initialization="he_normal",b_initialization=0,padding="SAME",
                     activation="relu",batch_norm=True,regularizer="L2",name="conv8_2")
    #conv8 = dropout(conv8,0.3)
    up9 = upsampling(conv8,[2,2])
    up9 = tf.concat(3,[up9,conv1])
    conv9 = conv_layer(up9,[3,3],32,
                     w_initialization="he_normal",b_initialization=0,padding="SAME",
                     activation="relu",batch_norm=True,regularizer="L2",name="conv9_1")
    conv9 = conv_layer(conv9,[3,3],32,
                     w_initialization="he_normal",b_initialization=0,padding="SAME",
                     activation="relu",batch_norm=True,regularizer="L2",name="conv9_2")
    conv10 = conv_layer(conv9,[1,1],1,
                        w_initialization="he_normal",b_initialization=0,padding="SAME",
                        activation="sigmoid",batch_norm=True,regularizer="L2",name="conv10")
    return conv10
