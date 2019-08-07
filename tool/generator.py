import sys


import threading
import os
import glob
import numpy as np 
import cv2
import config
import traceback
from tool.utils import BatchIndices


class Generator():
    def __init__(self,dir,batch_size = 2 , istraining = True,num_classes = 2,
                 trans_color = True,trans_gray = True,mirror=False,scale=True,clip=True,reshape=(640,640)):
        self.dir = dir 
        self.lock = threading.Lock()
        self.batch_size = batch_size
        self.shuffle =  istraining
        self.num_classes = num_classes
        self.mirror = mirror
        self.scale = scale
        self.reshape = reshape  #(h,w)
        self.clip = clip
        self.trans_color = trans_color
        self.trans_gray = trans_gray
        self.imagelist,self.labellist = self.list_dir(self.dir)
        self.batch_idx = BatchIndices(self.imagelist.shape[0],self.batch_size,self.shuffle)
    def num_classes(self):
        return self.num_classes

    def num_samples(self):
        return len(self.imagelist)

    def list_dir(self,dir):

        image =[]
        npy =[]

        imagesfile = glob.glob(os.path.join(dir,'*.jpg'))
        for i in imagesfile:
            npyfile = os.path.join(dir,'.'.join(os.path.basename(i).split('.')[:-1])+'.npy')
            imagefile = os.path.join(dir,i)
            if(os.path.exists(npyfile)):
                image.append(imagefile)
                npy.append(npyfile)
                
        return np.array(image),np.array(npy)

    def rand(self,a=0, b=1):
        return np.random.rand()*(b-a) + a

    def reshape_image(self,img,label,shape):
        lreshape = (int(shape[0]/config.ns),int(shape[1]/config.ns))
        lns = np.zeros((lreshape[0],lreshape[1],config.n))
        for c in range(config.n):
            lns[:,:,c] =cv2.resize(label[:,:,c],(lreshape[1],lreshape[0]),interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img,(self.reshape[1],self.reshape[0]),interpolation=cv2.INTER_AREA)
        return img,lns

    def scale_image(self,img,label,scalex,scaley):
        '''
        缩放并保证短边最少是640
        '''
        h,w = img.shape[0:2]
        h = int(h*scaley)
        w = int(w*scalex)

        h = max(h,self.reshape[0])
        w = max(w,self.reshape[1])

        lns = np.zeros((h,w,config.n))
        for c in range(config.n):
           lns[:,:,c] =cv2.resize(label[:,:,c],(w,h),interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)
        return img,lns

    def trans_color_image(self,img):
        '''
        颜色通道转换
        '''
        img = img[:,:,::-1]
        return img

    def trans_gray_image(self,img):
        img =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img =  cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        return img 

    def clip_image(self,img,label,shape):
        h,w = img.shape[0:2]
        ih,iw = shape 

        #img的短边要大于 shape的长边，不足的padding
        dh = max(h,ih)
        dw = max(w,iw)
        newimg = np.ones((dh,dw,img.shape[2]))*128
        newlabel = np.zeros((dh,dw,label.shape[2]))
        ty = (dh - h )//2
        tx = (dw - w)//2
        newimg[ty:ty+h,tx:tx+w,:] = img
        newlabel[ty:ty+h,tx:tx+w,:] = label
        h,w = (dh,dw)

        cx1,cy1,cx2,cy2=(0,0,0,0)
        for i in range(1000):
            cx1 = np.random.randint(0,w-iw+1)
            cy1 = np.random.randint(0,h-ih+1)
            cx2 = cx1 + iw 
            cy2 = cy1 + ih 

            #剪切到的文本面积过小则再随机个位置
            l = newlabel[cy1:cy2,cx1:cx2,-1]
            if(np.count_nonzero(l==1)>config.data_gen_clip_min_area):
                break

        img = newimg[cy1:cy2,cx1:cx2,:]
        label = newlabel[cy1:cy2,cx1:cx2,:]
        return img,label


    def __next__(self):
        idx = next(self.batch_idx)
        try:
            images = []
            labels = []
            for i,j in zip(self.labellist[idx],self.imagelist[idx]):
                l = np.load(i).astype(np.uint8)
                img = cv2.imread(j)
                #随机缩放
                if(self.scale):
                    scale = self.rand(config.data_gen_min_scales,config.data_gen_max_scales)
                    scalex = self.rand(scale-config.data_gen_itter_scales,scale+config.data_gen_itter_scales)
                    scaley = self.rand(scale-config.data_gen_itter_scales,scale+config.data_gen_itter_scales)
                    img,l = self.scale_image(img,l,scalex,scaley)

                #随机剪切
                if(self.clip):
                    img,l = self.clip_image(img,l,self.reshape)
                
                #颜色通道转换
                if(self.trans_color and np.random.randint(0,10)>5):
                    img = self.trans_color_image(img)

                if(self.trans_gray and np.random.randint(0,10)>7):
                    img = self.trans_gray_image(img)

                #reshape到训练尺寸
                if(self.reshape):
                    img,l = self.reshape_image(img,l,self.reshape)
                images.append(img)
                labels.append(l)

            images = np.array(images)
            labels = np.array(labels)
        
            seed = np.random.randint(0,100)


            if(self.mirror and  seed >90):
                images = images[:,::-1,::-1,:]
                labels = labels[:,::-1,::-1,:]
            elif(self.mirror and seed > 80):
                images = images[:,::-1,:,:]
                labels = labels[:,::-1,:,:]
            elif(self.mirror and seed > 70):
                images = images[:,:,::-1,:]
                labels = labels[:,:,::-1,:]
                
            return images, labels
        except Exception as e :
            print(e,j)
            traceback.print_exc()
            self.__next__()

##def test():
#gen = Generator(config.MIWI_2018_TEST_LABEL_DIR)

#images,labels = next(gen)
#print('images.shape',images.shape)
#print('labels.shape',labels.shape)
#import matplotlib.pyplot as plt 

#plt.imshow(images[1][:,:,::-1])

#plt.imshow(labels[0][:,:,5])


#z0 = np.count_nonzero(labels==0)
#z1 = np.count_nonzero(labels==1)
#print(z0+z1 == 2 * 320 * 320 * 6)

#test()
