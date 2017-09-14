
from nn.layers import *
def get_module(inp,num_classes):
    conv1 = conv_layer(inp,[7,7],64,2,padding=[3,3],name="conv1")
    pool1 = max_pool(conv1,[3,3],strides=[2,2],name="pool1")
    conv2 = conv_layer(pool1,[1,1],64,1,name="conv2")
    conv3 = conv_layer(conv2,[3,3],192,1,padding=[1,1],name="conv3")
    pool2 = max_pool(conv3,[3,3],strides=[2,2],name="pool2")
    inceptionA1 = inceptionFactoryA(pool2,64,64,64,64,96,"avg",32,"3a")
    inceptionA2 = inceptionFactoryA(inceptionA1,64,64,96,64,96,"avg",64,"3b")
    inceptionA3 = inceptionFactoryB(inceptionA2,128,160,64,96,"3c")
    inceptionA4 = inceptionFactoryA(inceptionA3,224,64,96,96,128,"avg",128,"4a")
    inceptionA5 = inceptionFactoryA(inceptionA4,192,96,128,96,128,"avg",128,"4b")
    inceptionA6 = inceptionFactoryA(inceptionA5,160,128,160,128,160,"avg",128,"4c")
    inceptionA7 = inceptionFactoryA(inceptionA6,96,128,192,160,192,"avg",128,"4d")
    inceptionA8 = inceptionFactoryB(inceptionA7,128,192,192,256,"4e")
    inceptionA10 = inceptionFactoryA(inceptionA8,352,192,320,160,224,"avg",128,"5a")
    inceptionA11 = inceptionFactoryA(inceptionA10,352,192,320,192,224,"max",128,"5b")
    avg1 = avg_pool(inceptionA11,[7,7],strides=[1,1],name="global_pool")
    x =flatten(avg1)
    fc12 = fc(x,num_classes,activation=None,name="fc12")
    return fc12



