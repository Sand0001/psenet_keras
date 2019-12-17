
max_depth = 256
mode = 'concat'
upsample_filters = {'add':[256,256,256,256],'concat':[128,128,128,128]}  #from buttom to top 
SN = 6           # number of kernel scales

rate_lc_ls = 0.7    #balances the importance between Lc and Ls 
rate_ohem = 3       #positive ：negtive = 1:rate_ohem

m = 0.5  #the minimal scale ration , which the value is (0,1]
n = 6   #the number of kernel scales

ns = 2  #“1s”, “2s” and “4s” means the width and height of the output map are 1/1, 1/2 and 1/4 of the input

visiable_gpu = '2,5'

MTWI_2018_TXT_DIR = r'/data/mahuichao/PSENET/data/MTWI_2018/label'
MTWI_2018_IMG_DIR = r'/data/mahuichao/PSENET/data/MTWI_2018/image'
MIWI_2018_TRAIN_LABEL_DIR = r'/data/mahuichao/PSENET/data/MTWI_2018/train_label'
MIWI_2018_TEST_LABEL_DIR = r'/data/mahuichao/PSENET/data/MTWI_2018/test_label'

# DIP_IMG_DIR = r'/data/mahuichao/PSENET/data/TEXT_SHENGCHAN190724'
# DIP_JSON_DIR = r'/data/mahuichao/PSENET/data/TEXT_SHENGCHAN190724'
# DIP_TRAIN_LABEL_DIR = r'/data/mahuichao/PSENET/data/TEXT_SHENGCHAN190724/train'
# DIP_TEST_LABEL_DIR = r'/data/mahuichao/PSENET/data/TEXT_SHENGCHAN190724/test'

DIP_IMG_DIR_TEXT_ZIXUAN = r'/data/mahuichao/PSENET/data/text_zixuan/image'
DIP_JSON_DIR_TEXT_ZIXUAN = r'/data/mahuichao/PSENET/data/text_zixuan/label'
DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN = r'/data/mahuichao/PSENET/data/text_zixuan/train'
DIP_TEST_LABEL_DIR_TEXT_ZIXUAN = r'/data/mahuichao/PSENET/data/text_zixuan/test'


DIP_IMG_DIR_TEXT_ZIXUAN2 = r'/data/mahuichao/PSENET/data/text_zixuan2/image'
DIP_JSON_DIR_TEXT_ZIXUAN2 = r'/data/mahuichao/PSENET/data/text_zixuan2/label'
DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN2 = r'/data/mahuichao/PSENET/data/text_zixuan2/train'
DIP_TEST_LABEL_DIR_TEXT_ZIXUAN2 = r'/data/mahuichao/PSENET/data/text_zixuan2/test'

DIP_IMG_DIR_TEXT_ZIXUAN3 = r'/data/mahuichao/PSENET/data/text_zixuan3/image'
DIP_JSON_DIR_TEXT_ZIXUAN3 = r'/data/mahuichao/PSENET/data/text_zixuan3/label'
DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN3 = r'/data/mahuichao/PSENET/data/text_zixuan3/train'
DIP_TEST_LABEL_DIR_TEXT_ZIXUAN3 = r'/data/mahuichao/PSENET/data/text_zixuan3/test'

DIP_IMG_DIR_TEXT_ZIXUAN4 = r'/data/mahuichao/PSENET/data/text_zixuan4/image'
DIP_JSON_DIR_TEXT_ZIXUAN4 = r'/data/mahuichao/PSENET/data/text_zixuan4/label'
DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN4 = r'/data/mahuichao/PSENET/data/text_zixuan4/train'
DIP_TEST_LABEL_DIR_TEXT_ZIXUAN4 = r'/data/mahuichao/PSENET/data/text_zixuan4/test'

###这是seal 数据
DIP_IMG_DIR_TEXT_ZIXUAN5 = r'/data/mahuichao/PSENET/data/text_zixuan5/image'
DIP_JSON_DIR_TEXT_ZIXUAN5 = r'/data/mahuichao/PSENET/data/text_zixuan5/label'
DIP_TRAIN_LABEL_DIR_TEXT_ZIXUAN5 = r'/data/mahuichao/PSENET/data/text_zixuan5/train'
DIP_TEST_LABEL_DIR_TEXT_ZIXUAN5 = r'/data/mahuichao/PSENET/data/text_zixuan5/test'



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


