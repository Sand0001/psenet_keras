
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
#with tf.device('/cpu:0'):
model  = keras.models.Model(inputs,output)
model.summary()
#%%
from keras.optimizers import Adam
from models.loss import build_loss
from models.metrics import build_iou,mean_iou
from keras.utils import multi_gpu_model

# model.load_weights('./tf/single0929.hdf5')

#%%
adam = Adam(1e-4)


#%%
ious = build_iou([0,1],['bk','txt'])

multi_model = multi_gpu_model(model)
#%%
multi_model.compile(loss=build_loss,
              optimizer=adam,
              metrics=ious)


#%%
from tool.generator import Generator

#%%
import config 
train_dir = [config.MIWI_2018_TRAIN_LABEL_DIR,
             config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN,
             config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN2,
             config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN3,
             config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN4,
             config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN,
             config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN2,
             config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN3,
             config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN4,
             config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN,
             config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN2,
             config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN3,
             config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN4,
             config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN,
             config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN2,
             config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN3,
             config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN4,
             config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN,
             config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN2,
             config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN3,
             config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN4,

                config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN5,
                config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN6,
             ]

test_dir = [config.MIWI_2018_TEST_LABEL_DIR,
            config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN,
            config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN2,
            config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN3,
            config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN4,
            config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN5,
            config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN6,]
batch_size = 8
num_class = 2 
shape = (640,640)


#%%
gen_train = Generator(train_dir,batch_size = batch_size ,istraining=True,
                        num_classes=num_class,mirror = False,reshape=shape,
                        trans_color = True,trans_gray=True,scale = True,
                        clip = True,angle = 10,max_size = None)

#%%
gen_test = Generator(test_dir,batch_size = 1 ,istraining = False,
                    num_classes=num_class,mirror = False,reshape=None,
                    trans_color = False,trans_gray=False,scale = False,
                    clip = False,angle = None,max_size = 1280)


#%%
from keras.callbacks import ModelCheckpoint,TensorBoard,LearningRateScheduler
checkpoint = ModelCheckpoint(r'./tf/finetune-{epoch:02d}.hdf5',
                           save_weights_only=True)
tb = TensorBoard(log_dir='./logs')

def schedule(epoch):
    if(epoch < 60):
        return 1e-4
    elif(epoch < 120):
        return 1e-5
    else:
        return 1e-6
lr = LearningRateScheduler(schedule)
#%%
res = multi_model.fit_generator(gen_train,
                          steps_per_epoch =gen_train.num_samples()// batch_size ,
                          epochs = 300,
                          validation_data=gen_test,
                          validation_steps =gen_test.num_samples()//batch_size ,
                          verbose=1,
                          initial_epoch=150,
                          workers=4,
                          use_multiprocessing=False,
                          max_queue_size=64,
                          callbacks=[tb,checkpoint,lr])

