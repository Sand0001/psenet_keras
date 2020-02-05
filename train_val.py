
import os
import config
os.environ['CUDA_VISIBLE_DEVICES'] = config.visiable_gpu
import cv2
import numpy as np
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
train_dir = [#config.MIWI_2018_TRAIN_LABEL_DIR,
#              config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN,
#              config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN2,
#              config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN3,
#              config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN4,
#              config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN,
#              config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN2,
#              config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN3,
#              config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN4,
#              config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN,
#              config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN2,
#              config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN3,
#              config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN4,
#              config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN,
#              config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN2,
#              config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN3,
#              config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN4,
#              config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN,
#              config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN2,
#              config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN3,
             config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN4,

                # config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN5,
                # config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN6,
             ]

test_dir = [#config.MIWI_2018_TEST_LABEL_DIR,
#             config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN,
#             config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN2,
#             config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN3,
            config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN4,
#             config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN5,
#             config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN6,
    ]
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

from sklearn.metrics import roc_auc_score


class Val_callback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(len(self.validation_data))
        print(self.validation_data)


class roc_callback(keras.callbacks.Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def pre_pic(self,images):
        MIN_LEN = 32
        MAX_LEN = 2200
        h, w = images.shape[0:2]
        if (w < h):
            if (w < MIN_LEN):
                scale = 1.0 * MIN_LEN / w
                h = h * scale
                w = MIN_LEN
            elif (h > MAX_LEN):
                scale = 1.0 * MAX_LEN / h
                w = w * scale if w * scale > MIN_LEN else MIN_LEN
                h = MAX_LEN
        elif (h <= w):
            if (h < MIN_LEN):
                scale = 1.0 * MIN_LEN / h
                w = scale * w
                h = MIN_LEN
            elif (w > MAX_LEN):
                scale = 1.0 * MAX_LEN / w
                h = scale * h if scale * h > MIN_LEN else MIN_LEN
                w = MAX_LEN

        w = int(w // 32 * 32)
        h = int(h // 32 * 32)

        scalex = images.shape[1] / w
        scaley = images.shape[0] / h

        images = cv2.resize(images, (w, h), cv2.INTER_AREA)
        images = np.reshape(images, (1, h, w, 3))
        return images,scalex,scaley


    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)

        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)

        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)), str(round(roc_val, 4))), end=100 * ' ' + '\n')
        return

val_callback = Val_callback()



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
                          callbacks=[tb,checkpoint,lr,val_callback])

