import tensorflow as tf

from layers import Dense, Conv2D, Flatten, Conv2DBatchNorm, MaxPool, Dropout, Activation

def VGG16(classes):
    return [
        Conv2D([3, 3], 64, [1, 1, 1, 1], padding='SAME',name="conv1_1"),
        
        Activation(tf.nn.elu),

        Conv2D([3, 3], 64, [1, 1, 1, 1], padding='SAME',name="conv1_2"),
    
        Activation(tf.nn.elu),

        MaxPool(ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME"),

        Conv2D([3, 3], 128, [1, 1, 1, 1],name="conv2_1"),
        
        Activation(tf.nn.elu),

        Conv2D([3, 3], 128, [1, 1, 1, 1], padding='SAME',name="conv2_2"),
        
        Activation(tf.nn.elu),
        MaxPool(ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME"),

        Conv2D([3, 3], 256, [1, 1, 1, 1],padding="SAME",name="conv3_1"),
        
        Activation(tf.nn.elu),

        Conv2D([3, 3], 256, [1, 1, 1, 1], padding='SAME',name="conv3_2"),
        
        Activation(tf.nn.elu),
        Conv2D([3, 3], 256, [1, 1, 1, 1], padding='SAME',name="conv3_3"),
        
        Activation(tf.nn.elu),
        MaxPool(ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME"),
        Conv2D([3, 3], 512, [1, 1, 1, 1],padding="SAME",name="conv4_1"),
        
        Activation(tf.nn.elu),

        Conv2D([3, 3], 512, [1, 1, 1, 1], padding='SAME',name="conv4_2"),
        
        Activation(tf.nn.relu),
        Conv2D([3, 3], 512, [1, 1, 1, 1], padding='SAME',name="conv4_3"),
        
        Activation(tf.nn.relu),
        MaxPool(ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME"),
         
        Conv2D([3, 3], 512, [1, 1, 1, 1],padding="SAME",name="conv5_1"),
        
        Activation(tf.nn.elu),

        Conv2D([3, 3], 512, [1, 1, 1, 1], padding='SAME',name="conv5_2"),
        
        Activation(tf.nn.elu),
        Conv2D([3, 3], 512, [1, 1, 1, 1], padding='SAME',name="conv5_3"),
        
        Activation(tf.nn.elu),
        MaxPool(ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME"),
        Flatten(),

        Dense(4096),
        Activation(tf.nn.elu),

        Dropout(0.5),

        Dense(4096),
        Activation(tf.nn.elu),
s
        Dropout(0.5),
        
        Dense(classes),
        #Activation(tf.nn.softmax),
        
    ]
