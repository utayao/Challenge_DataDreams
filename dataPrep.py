import cv2
import pdb
import os
import sys
import PIL
import h5py
import sys
import glob
import numpy as np
import time 
#import nn.image_processing as ip
from tqdm import tqdm
import initialization
from sklearn.utils import shuffle 
from PIL import Image
import tensorflow as tf
import utils
import logging 

flags = tf.app.flags
FLAGS = flags.FLAGS 
CLASSES = initialization.classes
flags.DEFINE_integer("size",224,"reduced dimension")
flags.DEFINE_bool("subset",True,"subset")
flags.DEFINE_string("train_path","/home/ashwin/Challenge_DataDreams/data","train file")
flags.DEFINE_integer("scaled_size",224,"scaled height")


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(image,label,image_name):
    index=[]
    #pdb.set_trace()
    save_class = None
    for i,l in enumerate(label):
        if l==1:
            index.append(i)
    if len(index)==0:
        save_class =  "none"
    else:
        save_class = "_".join(CLASSES[i] for i in index)
    save_path = os.path.join("train_images",save_class)
    makedirs(save_path)
    Image.fromarray(image).save(
                os.path.join(save_path,os.path.basename(image_name)))

def stack_imgs(images):
    Images = []
    Labels = []
    home_path = "/home/ashwin/MICCAI/detector/preprocess/train_images/"
    for i,img in tqdm(enumerate(images),total=len(images)):
        splits = img.split("/")
        img_name = splits[-1]
        #pdb.set_trace()
        label = Image.open(img).convert("1")
        #label = np.array(label).astype(int)
        #label = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
        
        label = label.resize((FLAGS.scaled_size,FLAGS.scaled_size),PIL.Image.ANTIALIAS)
        label = np.array(label).astype(int)
       
        #_,label = cv2.threshold(label,127,255,cv2.THRESH_BINARY)
        #pdb.set_trace()
        label = label.reshape((224,224,1))
        #print label.shape
        #print label.max()
        class_name = splits[-2]
        img_path = os.path.join(os.path.join(home_path,class_name),img_name)
        image = cv2.imread(img_path)
        
        #print image.shape
        #pdb.set_trace()
        image = cv2.resize(image,(FLAGS.scaled_size,FLAGS.scaled_size))
        #image = np.expand_dims(image,2)
        #image = ip.clahe(image)
        #pdb.set_trace()
        #image = np.array(image)
        #label = np.array(label)
        Images.append(image)
        Labels.append(label)
        #print img
        #print img_path
    
    Images = np.array(Images)
    Labels = np.array(Labels)
    #Image.fromarray(Labels[0,...]).show()
    h5 = h5py.File("train_unet1.h5")
    h5.create_dataset("images",data=Images)
    h5.create_dataset("labels",data=Labels)
    h5.close()
    return Images,Labels


        



def display_video(path):
    cap = cv2.VideoCapture(path)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    while(1):
        ret,frame = cap.read()
        fgmask = fgbg.apply(frame)
        cv2.imshow("frame",fgmask)
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
    cap.release()
    cv2.destroyAllWindows()
def extract_and_store(paths,files):
	h5=h5py.File(FLAGS.train_path,"a")
        #pdb.set_trace()

	start = True
	h5_images = None 
	h5_labels = None
	training_count =0
	subset=True
        Images = []
        Labels = []
        size = 30000
        count = 0
        index = 1
	for path,File in zip(paths,files):
            print "Current training count {}".format(training_count)
            print " reading from file {}".format(File)
            path = sorted(glob.glob(path+"/*"))
            with open(File,"r") as f:
                lines = f.readlines()
            if FLAGS.subset:
                lines = lines[:100]
            #print "total images in the file: ",len(lines[1:])
            for line in lines[1:]:
                
                line = line.strip().split("\t")
                image_index = int(line[0])
                image = Image.open(path[image_index])
                image = image.resize((FLAGS.size,FLAGS.size),PIL.Image.ANTIALIAS)

                image = np.array(image)
                label = [int(j) for j in line[1:]]
                Images.append(image)
                Labels.append(label)
                count +=1
                save_image(image,label,path[image_index])
                if count%size == 0:
                    #pdb.set_trace()
                    g1=h5.create_group("train_%s"%str(index))
                    Images = np.array(Images)
                    Labels = np.array(Labels)
                    Images,Labels = shuffle(Images,Labels)
                    g1.create_dataset("images",data=Images,compression="gzip",compression_opts=9)
                    g1.create_dataset("labels",data=Labels,compression="gzip",compression_opts=9)
                    print " Saving images and labels into group train_{} with image shape {} and label shape {}".format(index,Images.shape,Labels.shape)
                    index+=1
                    Images = []
                    Labels = []
                training_count+=1
        
        if len(Labels)>0:
             g1 = h5.create_group("train_%s"%str(index))
             Images = np.array(Images)
             Labels = np.array(Labels) 
             Images,Labels = shuffle(Images,Labels)
             g1.create_dataset("images",data=Images,compression="gzip",compression_opts=9)
             g1.create_dataset("labels",data=Labels,compression="gzip",compression_opts=9)                                                          
             print " Saving images and labels into group train_{} with image shape {} and label shape {}".format(index,Images.shape,Labels.shape)   
             index+=1
             Images = []
             Labels = []
             training_count +=1


def loadData(path):
    h5 = h5py.File(path,"r")
    return h5["images"],h5["labels"]

#def load_data(path):
#    h5 = h5py.File(path,"r")
#    x_train = h5.get("train_1")["images"]
#    y_train = h5.get("train_1")["labels"]
#    #x_train,y_train = batch_resize(x_train,y_train)
#    return x_train,y_train

def batch_resize(x_train,y_train):
    Images = []
    Labels = []
    print "x_train shape {} y_train shape {}".format(x_train.shape,y_train.shape)
    for img,label in tqdm(zip(x_train,y_train),total=x_train.shape[0]):
        img = images_processing.resize(img,(FLAGS.scaled_size,FLAGS.scaled_size))
        label = images_processing.resize(label,(FLAGS.scaled_size,FLAGS,scaled_size))
        Images.append(img)
        Labels.append(label)
    Images = np.array(Images)
    Labels = np.array(Labels)
    return Images,Labels


def display(epoch,**kwargs):
    
    backup = sys.stdout
    sys.stdout = LogToFile()
    print "#"*50
    print "===> Epoch: {}".format(epoch)
    for name,value in kwargs.items():
        print "{}={}".format(name,value)
    print "#"*50
    sys.stdout = backup

class LogToFile(object):
    def write(self,s):
        sys.__stdout__.write(s)
        open("%s.log"%(FLAGS.model_name),"a").write(s)
        
def load_data(path):
#    pdb.set_trace()
    images_path = glob.glob(os.path.join(path, 'patches/*.jpg'))
    images_arr = []
    labels_arr = []
    if FLAGS.subset:
        images_path = images_path[:50]
    for image_path in images_path:
        label = int(os.path.basename(image_path).replace('.jpg','').split('_')[-1])
        images_arr.append(utils.read_image(image_path))
        labels_arr.append(label)
    return np.array(images_arr), utils.one_hot_vector(labels_arr)





	


if __name__ == '__main__':

	#server = bc.ClientConnect("ashwinraju101@gmail.com","Tom9884552255^","+16822485471@mailmymobile.net")
	#server.startMonitor()	
	#paths =sorted(glob.glob("/media/ashwin/Radhika1TB/Dataset/tool_detection/25fps/*"))
	#files = sorted(glob.glob("/media/ashwin/Radhika1TB/Dataset/tool_detection/25fps_text/*.txt"))
	#extract_and_store(paths,files)
        #display_video("/media/ashwin/Radhika1TB/Dataset/tool_detection/tool_challenge_m2cai2016/tool_video_01.mp4")
        paths = glob.glob("/home/ashwin/MICCAI/detector/preprocess/train_masks_bw/*/*")
        #print paths
        stack_imgs(paths)
