
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


model.load_weights('./tf/resnet50.hdf5')
#%%
from keras.optimizers import Adam
from models.loss import build_loss
from models.metrics import build_iou,mean_iou
from keras.utils import multi_gpu_model


#%%
parallel_model = multi_gpu_model(model)


#%%
adam = Adam(1e-5)


#%%
ious = build_iou([0,1],['bk','txt'])


#%%
parallel_model.compile(loss=build_loss,
              optimizer=adam,
              metrics=ious)


#%%
from tool.generator import Generator

#%%
import config 
train_dir = config.DIP_TRAIN_LABEL_DIR
test_dir = config.DIP_TEST_LABEL_DIR
batch_size = 6
num_class = 2 
shape = (640,640)


#%%
gen_train = Generator(train_dir,batch_size = batch_size ,istraining=True,num_classes=num_class,mirror = False,reshape=shape,trans_color = False)

#%%
gen_train = Generator(train_dir,batch_size = batch_size ,istraining=True,num_classes=num_class,mirror = False,reshape=shape,trans_color = False)


#%%
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
checkpoint = ModelCheckpoint(r'./tf/finetune-{epoch:02d}.hdf5',
                           save_weights_only=True)
tb = TensorBoard(log_dir='./logs', update_freq=10)

#%%
res = parallel_model.fit_generator(gen_train,
                          steps_per_epoch =gen_train.num_samples()// batch_size,
                          epochs = 40,
#                          validation_data=gen_test,
 #                         validation_steps =gen_test.num_samples()//batch_size,
                          verbose=1,
                          initial_epoch=0,
                          workers=4,
                          use_multiprocessing=True,
                          max_queue_size=16,
                          callbacks=[tb])

res = parallel_model.fit_generator(gen_train,
                          steps_per_epoch =gen_train.num_samples()// batch_size,
                          epochs = 200,
#                          validation_data=gen_test,
 #                         validation_steps =gen_test.num_samples()//batch_size,
                          verbose=1,
                          initial_epoch=40,
                          workers=4,
                          use_multiprocessing=True,
                          max_queue_size=16,
                          callbacks=[checkpoint,tb])
