
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


#%%
from keras.optimizers import Adam
from models.loss import build_loss
from models.metrics import build_iou,mean_iou
from keras.utils import multi_gpu_model


#%%
parallel_model = multi_gpu_model(model)
parallel_model.load_weights('./tf/finetune-50.hdf5')


import cv2
images = cv2.imread('tmp.jpg')
images = np.reshape(images,(1,h,w,3))
res = model.predict(images[0:1,:,:,:])
res1 = res[0]
res1[res1>0.5]= 1
res1[res1<=0.5]= 0
res1 = (res1[:,:,5]*255).astype('uint8')
cv2.imwrite('./res.jpg',res1)

model.save_weights('./tf/signle_fintetune-50.hdf5')
