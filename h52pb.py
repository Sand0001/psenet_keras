
import os
import config
import sys
from tensorflow.python.tools import freeze_graph
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
multi_model.load_weights(sys.argv[1])
model.save_weights(sys.argv[2])


saver = tf.train.Saver()
saver.save(KTF.get_session(),r'./tf/resnet50.ckpt')

# n = [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

freeze_graph.freeze_graph(input_checkpoint = './tf/resnet50.ckpt',
                          input_meta_graph = './tf/resnet50.ckpt.meta',
                          output_graph = sys.argv[3],
                          output_node_names = 'activation_55/Sigmoid',
                          clear_devices = True,
                          input_graph ='',
                          input_saver = '',
                          input_binary = True,
                          restore_op_name = 'save/restore_all',
                          filename_tensor_name = 'save/Const:0',
                          initializer_nodes = '')
