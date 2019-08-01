
import os 
import config 
os.environ['CUDA_VISIBLE_DEVICES'] = config.visiable_gpu

#%%
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   
session = tf.Session(config=config)
KTF.set_session(session)


#%%
import keras 
from models.psenet import psenet


#%%
shape = (None,None,3)


#%%
inputs = keras.layers.Input(shape=shape)
output = psenet(inputs)
model  = keras.models.Model(inputs,output)
model.summary()
from keras.utils import multi_gpu_model

parallel_model = multi_gpu_model(model)
parallel_model.load_weights('./tf/finetune-41.hdf5')

from tool.generator import Generator

#%%
import config 
train_dir = config.DIP_TRAIN_LABEL_DIR
test_dir = config.DIP_TEST_LABEL_DIR
batch_size = 1
num_class = 2 
shape = (640,640)
gen_train = Generator(train_dir,batch_size = batch_size ,istraining=True,num_classes=num_class,mirror = False,reshape=shape)
 
import cv2
for i in range(100):
    images,_ = next(gen_train)
    cv2.imwrite('./tmp/res{}.jpg'.format(i),images[0])

import cv2
import numpy as np 
res = model.predict(images[0:1,:,:,:])
res1 = res[0]
res1[res1>0.5]= 255
res1[res1<=0.5]= 0
print('res1 shape',res1.shape)
print('nozero:',np.count_nonzero(res1>0.5))
for i in range(6):
    tmp = (res1[:,:,i]).astype(np.uint8)
    cv2.imwrite('./res{}.jpg'.format(i),tmp)
cv2.imwrite('./src.jpg',images[0])
#model.save_weights('./tf/signle_fintetune-50.hdf5')
