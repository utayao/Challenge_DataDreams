from layers import *

def alexnet(num_classes):
    return [
            Conv2D([11,11],96,[1,4,4,1],padding="SAME",name="conv1"),
            Activation(LeakyReLU(alpha=1/3.0)),
            LRN(radius=2,alpha=2e-05,beta=0.75,bias=1.0),
            MaxPool(ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID"),
            Conv2D([5,5],512,[1,1,1,1],padding="SAME",name="conv2"),
            Activation(LeakyReLU(alpha=1/3.0)),
            LRN(radius=2,alpha=2e-05,beta=0.75,bias=1.0),
            MaxPool(ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID"),
            Conv2D([3,3],384,[1,1,1,1],name="conv3"),
            Activation(LeakyReLU(alpha=1/3.0)),
            Conv2D([3,3],768,[1,1,1,1],padding="SAME",name="conv4"),
            Activation(LeakyReLU(alpha=1/3.0)),
            MaxPool(ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID"),
            Conv2D([3,3],512,[1,1,1,1],padding="SAME",name="conv5"),
            Activation(LeakyReLU(alpha=1/3.0)),
        
            Flatten(),
            Dense(4096,name="fc6"),
            Activation(LeakyReLU(alpha=1/3.0)),
            Dropout(0.5),
            Dense(4096,name="fc7"),
            Activation(LeakyReLU(alpha=1/3.0)),
            Dropout(0.5),
            Dense(num_classes,name="fc8"),
            ]
