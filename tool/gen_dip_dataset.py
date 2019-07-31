import sys
import cv2
import pyclipper 
import os 
import traceback
sys.path.append(r'/Users/mahuichao/Documents/PSENET')
import shutil
import multiprocessing as mp
from itertools import repeat
import config
import numpy as np 
from tool.dip_data import read_dataset
from tool.utils import del_allfile , convert_label_to_id

def cal_di(pnt,m,n):
    '''
    calculate di pixels for shrink the original polygon pnt 
    Arg:
        pnt : the points of polygon [[x1,y1],[x2,y2],...]
        m : the minimal scale ration , which the value is (0,1]
        n : the number of kernel scales
    return di_n [di1,di2,...din] 
    '''

    area = cv2.contourArea(pnt)
    perimeter = cv2.arcLength(pnt,True)

    ri_n = [] 
    for i in range(1,n):
        ri = 1.0 - (1.0 - m) * (n - i) / (n-1)
        ri_n.append(ri)

    di_n = []
    for ri in ri_n:
        di = area * (1 - ri * ri ) / perimeter
        di_n.append(di)

    return di_n

 
def shrink_polygon(pnt,di_n):
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(pnt, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

    shrink_pnt_n = [] 
    for di in di_n:
        shrink_pnt = pco.Execute(-int(di))
        shrink_pnt_n.append(shrink_pnt)
    return shrink_pnt_n


def gen_dataset(data):
    imgname,gtboxes = data[0]
    dst_dir = data[1]
    try:
        basename = '.'.join(os.path.basename(imgname).split('.')[:-1])
        img = cv2.imread(imgname)

        MIN_LEN = 32
        MAX_LEN = 4680
        h, w = img.shape[0:2]
        if(w<h):
            if(w<MIN_LEN):
                scale = 1.0 * MIN_LEN / w
                h = h * scale 
                w = MIN_LEN
            elif(h>MAX_LEN):
                scale = 1.0 * MAX_LEN / h 
                w = w * scale if w * scale > MIN_LEN else MIN_LEN
                h = MAX_LEN
        elif(h<=w ):
            if(h<MIN_LEN):
                scale = 1.0 * MIN_LEN / h
                w = scale * w
                h = MIN_LEN 
            elif(w>MAX_LEN):
                scale = 1.0 * MAX_LEN / w
                h = scale * h if scale * h >  MIN_LEN else MIN_LEN
                w = MAX_LEN


        scalex = img.shape[1] / w
        scaley = img.shape[0] / h
        w , h = int(w) , int(h)
        img = cv2.resize(img,(w,h),cv2.INTER_AREA)


        labels = np.ones((config.n,img.shape[0],img.shape[1],3),np.uint8)
        labels = labels * 255
        npys = np.zeros((img.shape[0],img.shape[1],config.n),np.uint8)

        gtboxes = np.array(gtboxes)

        gtboxes[:,:,0] = gtboxes[:,:,0] / scalex
        gtboxes[:,:,1] = gtboxes[:,:,1] / scaley
        #shrink 1.0
        for gtbox in gtboxes:
            cv2.drawContours(labels[config.n-1],[gtbox],-1,(0,0,255),-1)
        

        #shrink n-1 times
        for gtbox in gtboxes:
            di_n = cal_di(gtbox,config.m,config.n)
            shrink_pnt_n = shrink_polygon(gtbox,di_n)
            for id,shirnk_pnt in enumerate(shrink_pnt_n):             
                cv2.drawContours(labels[id],np.array(shirnk_pnt),-1,(0,0,255),-1)
        
        for i in range(config.n):
            cv2.imwrite(os.path.join(dst_dir,basename+'_'+str(i)+'.tif'),labels[i])

        cv2.imwrite(os.path.join(dst_dir,basename+'.jpg'),img)

        #convert labelimage to id 
        for idx,label in enumerate(labels):
            npy = convert_label_to_id(config.label_to_id,label)
            npys[:,:,idx] = npy
        np.save(os.path.join(dst_dir,basename+'.npy'),npys)
    except Exception as e :
        print(imgname,traceback.print_exc())
        #ddd = r'E:\psenet-MTWI\document\mtwi_2018_train\bad'
        #shutil.copyfile(imgname,os.path.join(ddd,basename))

def create_dataset():
    data = read_dataset(config.DIP_JSON_DIR,config.DIP_IMG_DIR)

    #split trian and test data
    train_num = int(len(data) * 0.9)
    train_data = {key:data[key] for i,key in enumerate(data) if i<train_num }
    test_data  = {key:data[key] for i,key in enumerate(data) if i>=train_num }

    del_allfile(config.DIP_TRAIN_LABEL_DIR)
    gen_dataset(train_data,config.DIP_TRAIN_LABEL_DIR)

    del_allfile(config.DIP_TEST_LABEL_DIR)
    gen_dataset(test_data,config.DIP_TEST_LABEL_DIR)

if __name__=='__main__':
    data = read_dataset(config.DIP_JSON_DIR,config.DIP_IMG_DIR)
    #split trian and test data
    train_num = int(len(data) * 0.9)
    train_data = {key:data[key] for i,key in enumerate(data) if i<train_num }
    test_data  = {key:data[key] for i,key in enumerate(data) if i>=train_num }

    del_allfile(config.DIP_TRAIN_LABEL_DIR)
    del_allfile(config.DIP_TEST_LABEL_DIR)

    with mp.Pool(processes=1) as pool:
        pool.map(gen_dataset,zip(train_data.items(),repeat(config.DIP_TRAIN_LABEL_DIR)))
        pool.map(gen_dataset,zip(test_data.items(),repeat(config.DIP_TEST_LABEL_DIR)))


