from nn.layers import * 

def get_module(inp,num_classes):
    conv1 = conv_layer(inp,[3,3],32,
                        w_initialization="he_normal",b_initialization=0,padding="VALID",
                        activation="relu",batch_norm=False,name="conv1_1")
    conv1 = conv_layer(conv1,[3,3],64,
                        w_initialization="he_normal",b_initialization=0,padding="VALID",
                        activation="relu",batch_norm=False,name="conv1_2")

    pool1 = max_pool(conv1,[2,2],[2,2],name="max_pool1",padding="SAME")
    conv2 = conv_layer(pool1,[3,3],128,
                        w_initialization="he_normal",b_initialization=0,padding="VALID",
                        activation="relu",batch_norm=False,name="conv2_1")
    conv2 = conv_layer(conv2,[3,3],128,
                        w_initialization="he_normal",b_initialization=0,padding="VALID",
                        activation="relu",batch_norm=False,name="conv2_2")

    pool2 = max_pool(conv2,[2,2],[2,2],name="max_pool2",padding="SAME")

    conv3 = conv_layer(pool2,[3,3],256,
                        w_initialization="he_normal",b_initialization=0,padding="VALID",
                        activation="relu",batch_norm=False,name="conv3_1")
    conv3 = conv_layer(conv3,[3,3],256,
                        w_initialization="he_normal",b_initialization=0,padding="VALID",
                        activation="relu",batch_norm=False,name="conv3_2")

    pool3 = max_pool(conv3,[2,2],[2,2],name="max_pool3",padding="SAME")

    conv4 = conv_layer(pool3,[3,3],512,
                        w_initialization="he_normal",b_initialization=0,padding="VALID",
                        activation="relu",batch_norm=False,name="conv4_1")
    conv4 = conv_layer(conv4,[3,3],512,
                        w_initialization="he_normal",b_initialization=0,padding="VALID",
                        activation="relu",batch_norm=False,name="conv4_2")
    pool4 = max_pool(conv4,[2,2],[2,2],name="max_pool4",padding="SAME")

    conv5 = conv_layer(pool4,[3,3],1024,
                        w_initialization="he_normal",b_initialization=0,padding="VALID",
                        activation="relu",batch_norm=False,name="conv5_1")
    conv5 = conv_layer(conv5,[3,3],1024,
                        w_initialization="he_normal",b_initialization=0,padding="VALID",
                        activation="relu",batch_norm=False,name="conv5_2")
    up6_1 = upsampling(conv5,(2,2))
    up6 = conv_layer(up6_1,[2,2],512,2,w_initialization="he_normal",
                    b_initialization=0,padding="SAME",activation="relu",batch_norm=False,name="up6_conv")
    up6 = tf.concat(0,[upsampling(up6,(2,2)),conv4])
    conv6 = conv_layer(up6,[3,3],1024,
                        w_initialization="he_normal",b_initialization=0,padding="VALID",
                        activation="relu",batch_norm=False,name="conv6_1")
    conv6 = conv_layer(conv6,[3,3],512,
                        w_initialization="he_normal",b_initialization=0,padding="VALID",
                        activation="relu",batch_norm=False,name="conv6_2")
    up7 = tf.concat(3,[upsampling(conv6,(2,2)),conv6],mode="concat")

    conv7 = conv_layer(up7,[3,3],512,
                        w_initialization="he_normal",b_initialization=0,padding="VALID",
                        activation="relu",batch_norm=False,name="conv7_1")
    conv7 = conv_layer(conv7,[3,3],256,
                        w_initialization="he_normal",b_initialization=0,padding="VALID",
                        activation="relu",batch_norm=False,name="conv7_2")
    up8 = tf.concat(3,[upsampling(conv7,size=(2,2)),conv7])

    conv8 = conv_layer(up8,[3,3],256,
                        w_initialization="he_normal",b_initialization=0,padding="VALID",
                        activation="relu",batch_norm=False,name="conv8_1")

    conv8 = conv_layer(conv8,[3,3],128,
                     w_initialization="he_normal",b_initialization=0,padding="VALID",
                     activation="relu",batch_norm=False,name="conv8_1")
    up9 = tf.concat(3,[upsampling(conv8,size=(2,2)),conv8])

    conv10 = conv_layer(up9,[3,3],128,
                     w_initialization="he_normal",b_initialization=0,padding="VALD",
                     activation="relu",batch_norm=False,name="conv9_1")
    conv10 = conv_layer(conv10,[3,3],64,
                     w_initialization="he_normal",b_initialization=0,padding="VALID",
                     activation="relu",batch_norm=False,name="conv9_2")
    conv10 = conv_layer(conv10,64,[3,3],
                     w_initialization="he_normal",b_initialization=0,padding="VALID",
                     activation="relu",batch_norm=False,name="conv9_3")
    conv10 = conv_layer(conv10,[1,1],num_classes,
                     w_initialization="he_normal",b_initialization=0,padding="VALID",
                     activation=None,batch_norm=False,name="conv10_1")

    return conv10
