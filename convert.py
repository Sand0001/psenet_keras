
import os 
import config 
os.environ['CUDA_VISIBLE_DEVICES'] = '4,7'

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
#with tf.device('/cpu:0'):
model  = keras.models.Model(inputs,output)
model.summary()
#%%
from keras.optimizers import Adam
from models.loss import build_loss
from models.metrics import build_iou,mean_iou
from keras.utils import multi_gpu_model



#%%
adam = Adam(1e-4)


#%%
ious = build_iou([0,1],['bk','txt'])
multi_model = multi_gpu_model(model)
multi_model.load_weights('./tf/finetune-150.hdf5')
model.save_weights('single1009.hdf5')
