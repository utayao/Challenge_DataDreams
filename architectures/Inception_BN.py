
from layers import *
def Inception_BN(num_classes):
	return [
		ConvFactory([7,7],64,[1,2,2,1],[3,3],name="conv1"),
		MaxPool(ksize=[1,3,3,1],strides=[1,2,2,1],name="pool1"),
		ConvFactory([1,1],64,[1,1,1,1],name="conv2"),
		ConvFactory([3,3],192,[1,1,1,1],[1,1],name="conv3"),
		MaxPool(ksize=[1,3,3,1],strides=[1,2,2,1],name="pool2"),
		InceptionFactoryA(64,64,64,64,96,"avg",32,"3a"),
		InceptionFactoryA(64,64,96,64,96,"avg",64,"3b"),
		InceptionFactoryB(128,160,64,96,"3c"),
		InceptionFactoryA(224,64,96,96,128,"avg",128,"4a"),
		InceptionFactoryA(192,96,128,96,128,"avg",128,"4b"),
		InceptionFactoryA(160,128,160,128,160,"avg",128,"4c"),
		InceptionFactoryA(96,128,192,160,192,'avg',128,"4d"),
		InceptionFactoryB(128,192,192,256,"4e"),

		InceptionFactoryA(352,192,320,160,224,"avg",128,"5a"),
		InceptionFactoryA(352,192,320,192,224,"max",128,"5b"),
		AvgPool(ksize=[1,7,7,1],strides=[1,1,1,1],name="global_pool"),
		Flatten(),
		Dense(num_classes),
		#Activation(tf.nn.softmax),
	

	]
