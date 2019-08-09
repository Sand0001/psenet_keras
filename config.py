
max_depth = 256
upsample_filters = [256,256,256,256]  #from buttom to top 
SN = 6           # number of kernel scales

rate_lc_ls = 0.7    #balances the importance between Lc and Ls 
rate_ohem = 3       #positive ：negtive = 1:rate_ohem

m = 0.5  #the minimal scale ration , which the value is (0,1]
n = 6   #the number of kernel scales

ns = 2  #“1s”, “2s” and “4s” means the width and height of the output map are 1/1, 1/2 and 1/4 of the input

visiable_gpu = '1,5'

MTWI_2018_TXT_DIR = r'/data/mahuichao/PSENET/data/MTWI_2018/label'
MTWI_2018_IMG_DIR = r'/data/mahuichao/PSENET/data/MTWI_2018/image'
MIWI_2018_TRAIN_LABEL_DIR = r'/data/mahuichao/PSENET/data/MTWI_2018/train_label'
MIWI_2018_TEST_LABEL_DIR = r'/data/mahuichao/PSENET/data/MTWI_2018/test_label'

DIP_IMG_DIR = r'/data/mahuichao/PSENET/data/TEXT_SHENGCHAN190724'
DIP_JSON_DIR = r'/data/mahuichao/PSENET/data/TEXT_SHENGCHAN190724'
DIP_TRAIN_LABEL_DIR = r'/data/mahuichao/PSENET/data/TEXT_SHENGCHAN190724/train'
DIP_TEST_LABEL_DIR = r'/data/mahuichao/PSENET/data/TEXT_SHENGCHAN190724/test'

label_to_id = {(255,255,255):0,(0,0,255):1}

data_gen_min_scales = 0.8
data_gen_max_scales = 2
data_gen_itter_scales = 0.3

#随机剪切 文字区域最小面积
data_gen_clip_min_area = 20*100


#dice loss
batch_loss = True

#metric iou 
metric_iou_batch = True


