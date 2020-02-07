
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
from cal_recall import cal_recall_precison_f1
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

from psenet.utils_up4 import scale_expand_kernels ,fit_minarearectange
from psenet.utils_up4 import calc_vote_angle , fit_boundingRect_warp_cpp

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
             config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN2,
             config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN3,
             config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN4,
             ]

test_dir = [config.MIWI_2018_TEST_LABEL_DIR,
            config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN,
            config.DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN2,
            config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN3,
            config.DIP_TEST_LABEL_DIR_TEXT_ZIXUAN4,
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


class V1al_callback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print('len(self.validation_data)',len(self.validation_data))
        print('self.validation_data',self.validation_data)


class Val_callback(keras.callbacks.Callback):


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

    def post_process(self,pic,res,scalex,scaley,det_path):
        res1 = res[0]
        res1[res1 > 0.9] = 1
        res1[res1 <= 0.9] = 0
        newres1 = []
        print('res1.shape',res1.shape)

        for i in range(0, 5):
            n = np.logical_and(res1[:, :, 5], res1[:, :, i]) * 255
            n = n.astype('int32')
            newres1.append(n)

        # 计算角度
        degree = calc_vote_angle(newres1[-1])
        num_label, labelimage = scale_expand_kernels(newres1, filter=False)
        labelimage_tmp = labelimage.copy()
        labelimage_tmp[labelimage_tmp > 0] = 255
        h, w = labelimage.shape[0:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1.0)
        neg_M = cv2.getRotationMatrix2D((w / 2, h / 2), -degree, 1.0)
        rects = fit_minarearectange(num_label - 1, labelimage)
        rects = np.array(rects)
        if rects.shape[0] > 0:
            # print(rects.shape)
            rects = cv2.transform(np.array(rects), neg_M)
            # rects_a = rects.tolist()
        if rects.shape[0] > 0:
            rects = rects.reshape(-1, 8)

        results = []
        for rt in rects:
            # rt = [rt[0][0], rt[0][1], rt[1][0], rt[1][1], rt[2][0], rt[2][1], rt[3][0], rt[3][1]]
            rt[0] = rt[0] * 2 * scalex
            rt[1] = rt[1] * 2 * scaley
            rt[2] = rt[2] * 2 * scalex
            rt[3] = rt[3] * 2 * scaley
            rt[4] = rt[4] * 2 * scalex
            rt[5] = rt[5] * 2 * scaley
            rt[6] = rt[6] * 2 * scalex
            rt[7] = rt[7] * 2 * scaley
            # rt[4], rt[6] = rt[6], rt[4]
            # rt[5], rt[7] = rt[7], rt[5]
            # rt = np.append(rt, degree)
            results.append(rt)
        # np.save(os.path.join(det_path,pic), np.array(results))
        a = np.array(results)
        np.savetxt(os.path.join(det_path,pic) + '.txt', a, delimiter=',', fmt='%d')


    def on_epoch_end(self, epoch, logs={}):
        val_data_path = '/data/mahuichao/PSENET/data/det_pic'
        det_path = '/data/mahuichao/PSENET/data/det_txt'
        gt_path = '/data/mahuichao/PSENET/data/gt_txt'

        if epoch > 60 and epoch % 5 == 0:
            val_data_list = os.listdir(val_data_path)
            for pic in val_data_list:
                img = cv2.imread(os.path.join(val_data_path,pic))
                img,scalex,scaley = self.pre_pic(img)

                res = self.model.predict(img)
                self.post_process(pic,res,scalex,scaley,det_path)
            result_dict = cal_recall_precison_f1(gt_path, det_path)
            print('val_data',result_dict)
        return

val_callback = Val_callback()



#%%
res = multi_model.fit_generator(gen_train,
                          steps_per_epoch =gen_train.num_samples()// batch_size ,
                          epochs = 300,
                          validation_data=gen_test,
                          validation_steps =gen_test.num_samples()//batch_size ,
                          verbose=1,
                          initial_epoch=0,
                          workers=4,
                          use_multiprocessing=False,
                          max_queue_size=64,
                          callbacks=[tb,checkpoint,lr,val_callback])

