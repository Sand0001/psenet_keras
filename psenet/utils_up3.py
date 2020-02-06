import threading
import numpy as np 
import csv
import os
import shutil
import cv2
from sklearn.cluster import KMeans
import glob
from collections import Counter
from copy import deepcopy
from numba import jit
import sys
sys.path.append(os.getcwd() + '/psenet')
from pse import find_label_coord,ufunc_4_cpp

import tensorflow as tf 

class BatchIndices():
    def __init__(self,total,batchsize,trainable=True):
        self.n = total
        self.bs = batchsize
        self.shuffle = trainable
        self.lock = threading.Lock()
        self.reset()
    def reset(self):
        self.index = np.random.permutation(self.n) if self.shuffle==True else np.arange(0,self.n)
        self.curr = 0
    
    def __next__(self):
        with self.lock:
            if self.curr >=self.n:
                self.reset()
            rn = min(self.bs,self.n - self.curr)
            res = self.index[self.curr:self.curr+rn]
            self.curr += rn
            return res

def del_allfile(path):
    '''
    del all files in the specified directory
    '''
    filelist = glob.glob(os.path.join(path,'*.*'))
    for f in filelist:
        os.remove(os.path.join(path,f))



def convert_label_to_id(label2id,labelimg):
    '''
    convert label image to id npy
    param:
    labelimg - a label image with 3 channels
    label2id  - dict eg.{(0,0,0):0,(0,255,0):1,....}
    '''

    h,w = labelimg.shape[0],labelimg.shape[1]
    npy = np.zeros((h,w),'uint8')
    
    for i,j in label2id.items():
        idx = ((labelimg == i) * 1)
        idx = np.sum(idx,axis=2) >=3
        npy = npy + (idx * j).astype(np.uint8)
    return npy


def convert_id_to_label(id,label2id):
    '''
    convet id numpy to label image 
    param:
    id          : numpy
    label2id  - dict eg.{(0,0,0):0,(0,255,0):1,....}
    return labelimage 
    '''
    h,w = id.shape[0],id.shape[1]

    labelimage = np.ones((h,w,3),'uint8') * 255
    for i,j in label2id.items():
        labelimage[np.where(id==j)] = i 

    return labelimage
 

@jit
def ufunc_4(S1,S2,TAG):
    #indices 四邻域 x-1 x+1 y-1 y+1，如果等于TAG 则赋值为label
    for h in range(1,S1.shape[0]-1):
        for w in range(1,S1.shape[1]-1):
            label = S1[h][w]
            if(label!=0):
                if(S2[h][w-1] == TAG):                          
                    S2[h][w-1] = label
                if(S2[h][w+1] == TAG):                            
                    S2[h][w+1] = label
                if(S2[h-1][w] == TAG):                            
                    S2[h-1][w] = label
                if(S2[h+1][w] == TAG):                           
                    S2[h+1][w] = label
                    
def scale_expand_kernel(S1,S2):
    TAG = 10240                     
    S2[S2==255] = TAG
    mask = (S1!=0)
    S2[mask] = S1[mask]
    cond = True 
    while(cond):  
        before = np.count_nonzero(S1==0)
        ufunc_4_cpp(S1,S2,TAG)  
        S1[S2!=TAG] = S2[S2!=TAG]  
        after = np.count_nonzero(S1==0)
        if(before<=after):
            cond = False
       
    return S1

def filter_label_by_area(labelimge,num_label,area=5):
    for i in range(1,num_label+1):
        if(np.count_nonzero(labelimge==i)<=area):
            labelimge[labelimge==i] ==0
    return labelimge

def scale_expand_kernels(kernels,filter=False):
    '''
    args:
        kernels : S(0,1,2,..n) scale kernels , Sn is the largest kernel
    '''
    S = kernels[0]
    num_label,labelimage = cv2.connectedComponents(S.astype('uint8'))
    if(filter==True):
        labelimage = filter_label_by_area(labelimage,num_label)
    for Si in kernels[1:]:
        labelimage = scale_expand_kernel(labelimage,Si)
    return num_label,labelimage   

def fit_minarearectange(num_label,labelImage):
    rects= []
    for label in range(1,num_label+1):
        points = np.array(np.where(labelImage == label)[::-1]).T

        rect = cv2.minAreaRect(points)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        area = cv2.contourArea(rect)
        if(area<10):
            print('area:',area)
            continue
        rects.append(rect)
    return rects

def fit_minarearectange_cpp(num_label,labelimage):
    rects = [] 
    points = find_label_coord(labelimage,num_label)
    for i in range(num_label):
        pt = np.array(points[i]).reshape(-1,2)
        rect = cv2.minAreaRect(pt)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        rects.append(rect)
    
    return rects 



def order_points(pts):
    #https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    rects = [] 
    for pt in pts :
        xSorted = pt[np.argsort(pt[:,0]),:]
    
        leftMost = xSorted[:2,:]
        rightMost = xSorted[2:,:]
    
        leftMost = leftMost[np.argsort(leftMost[:,1]),:]
        (tl,bl) = leftMost

        # D  = distance.cdist(tl[np.newaxis],rightMost,'euclidean')[0]
        # (br,tr) = rightMost[np.argsort(D)[::-1],:]
        rightMost = rightMost[np.argsort(rightMost[:,1]),:]
        (tr,br) = rightMost

        rects.append(np.array([tl,tr,br,bl],dtype='int32'))
    
    return rects

def calc_vote_angle(bin_img):
    '''
    二值图进行骨架化处理后用houghline计算角度
    设定不同累加阈值（图像宽度的[4-6]分之一）多次计算投票确定最终角度
    '''
    def cal_angle(thin_img,threshold):
        lines = cv2.HoughLines(thin_img,1,np.pi/360,threshold)
        if(lines is None):
            return None
        angles = []
        for line in lines:
            rho,theta = line[0]
            ## 精度0.5
            angles.append(theta * 180 / np.pi //0.5 * 0.5)
        return Counter(angles).most_common(1)[0][0]
    
    thin_img = bin_img.astype(np.uint8)
    thin_img_w = thin_img.shape[1]
    thin_img = cv2.ximgproc.thinning(thin_img)
    angles =[]
    for ratio in [4,5,6]:
        angle = cal_angle(np.copy(thin_img),thin_img_w//ratio)
        if(angle == None):
            continue
        angles.append(angle)

    most_angle  = Counter(angles).most_common(1)  
    most_angle =  0 if len(most_angle)==0 else most_angle[0][0]

    if(most_angle>=0 and most_angle<=45):
        most_angle = most_angle - 90
    elif(most_angle>45 and most_angle<=90):
        most_angle = most_angle - 90
    elif(most_angle>90 and most_angle<=135):
        most_angle = most_angle - 90
    elif(most_angle>135 and most_angle<180):
        most_angle = most_angle - 90
    return most_angle


def save_MTWI_2108_resault(filename,rects,scalex=1.0,scaley=1.0):
    with open(filename,'w',encoding='utf-8') as f:
        for rect in rects:
            line = ''
            for r in rect:
                line += str(r[0] * scalex) + ',' + str(r[1] * scaley) + ','
            line = line[:-1] + '\n'
            f.writelines(line)

def fit_boundingRect(num_label,labelImage):
    rects= []
    for label in range(1,num_label+1):
        points = np.array(np.where(labelImage == label)[::-1]).T
        x,y,w,h = cv2.boundingRect(points)
        rect = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
        rects.append(rect)
    return rects


def fit_boundingRect_cpp(num_label,labelimage):
    rects = [] 
    points = find_label_coord(labelimage,num_label)
    for i in range(num_label):
        pt = np.array(points[i]).reshape(-1,2)
        x,y,w,h = cv2.boundingRect(pt)
        rects.append(np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]]))
    return rects

def fit_boundingRect_warp_cpp(num_label,labelimage,M):
    rects = [] 
    points = find_label_coord(labelimage,num_label)
    for i in range(num_label):
        pt = np.array(points[i]).reshape(1,-1,2)
        pt = cv2.transform(pt,M)
        x,y,w,h = cv2.boundingRect(pt)
        pt = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
        rects.append(pt)
    return rects

def warpAffine_Padded(src_h,src_w,M,mode='matrix'):
    '''
    重新计算旋转矩阵，防止cut off image
    args：
        src_h,src_w 原图的高、宽
        mode: mode is matrix 时 M 是旋转矩阵
              mode is angle 时  M 是角度
    returns:
        offset_M : 新的旋转矩阵
        padded_w,padded_h : 图像的新宽、高
    
    ------------------------------------
    用法：
        h,w = imagetest.shape[0:2]
        M = cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
        offset_M , padded_w , padded_h = warpAffinePadded(h,w,M)
        rects = cv2.transform(rects,offset_M)
        imagetest = cv2.warpAffine(imagetest,offset_M,(padded_w,padded_h))
    '''
    if(mode == 'angle'):
        M = cv2.getRotationMatrix2D((src_w/2,src_h/2),M,1.0)
    
    # 图像四个顶点
    lin_pts = np.array([
        [0,0],
        [src_w,0],
        [src_w,src_h],
        [0,src_h]
    ])
    trans_lin_pts = cv2.transform(np.array([lin_pts]),M)[0]
    
    #最大最小点
    min_x = np.floor(np.min(trans_lin_pts[:,0])).astype(int)
    min_y = np.floor(np.min(trans_lin_pts[:,1])).astype(int)
    max_x = np.ceil(np.max(trans_lin_pts[:,0])).astype(int)
    max_y = np.ceil(np.max(trans_lin_pts[:,1])).astype(int)

    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0
    #print('offsetx:{},offsety:{}'.format(offset_x,offset_y))
    offset_M = M + [[0,0,offset_x],[0,0,offset_y]]

    padded_w = src_w + (max_x - src_w)  + offset_x 
    padded_h = src_h + (max_y - src_h)  + offset_y 
    return offset_M,padded_w,padded_h

class text_porposcal:
    def __init__(self,rects,max_dist = 50 , scale_h = 1.5 ,threshold_overlap_v = 0.5):
        self.rects = np.array(rects) 
        #offset
        rects , max_w , offset = self.offset_coordinate(self.rects)
        self.rects = rects
        self.max_w = max_w
        self.offset = offset

        self.max_dist = max_dist 
        self.scale_h = scale_h
        self.threshold_overlap_v = threshold_overlap_v
        self.graph = np.zeros((self.rects.shape[0],self.rects.shape[0]))
        self.r_index = [[] for _ in range(self.max_w)]
        for index , rect in enumerate(rects):
            self.r_index[int(rect[0][0])].append(index)
        
        #记录已经参与textline的框
        self.tmp_connected = []

    def offset_coordinate(self,rects):
        '''
        经过旋转的坐标有时候被扭到了负数，你敢信？
        所以我们要算出最大的负坐标，然后上这个offset,在完成textline以后再减回去
        '''
        if(rects.shape[0] == 0 ):
            return rects , 0 , 0 

        offset = rects.min()
        max_w = rects[:,:,0].max() + 1 
        offset = - offset if offset < 0 else 0
        rects = rects + offset
        max_w = max_w + offset
        return rects , max_w , offset


    def get_sucession(self,index):
        rect = self.rects[index]
        #以高度作为搜索长度
        max_dist =  int((rect[3][1] - rect[0][1] ) * self.scale_h)
        max_dist = min(max_dist , self.max_dist)    
        for left in range(rect[0][0]+1,min(self.max_w-1,rect[1][0]+max_dist)):
            for idx in self.r_index[left]:
                iou = self.meet_v_iou(index,idx)
                if(iou > self.threshold_overlap_v):
                    return idx 
        return -1 

    def meet_v_iou(self,index1,index2):
        '''

        '''
        height1 = self.rects[index1][3][1] - self.rects[index1][0][1]
        height2 = self.rects[index2][3][1] - self.rects[index2][0][1]
        y0 = max(self.rects[index1][0][1],self.rects[index2][0][1])
        y1 = min(self.rects[index1][3][1],self.rects[index2][3][1])
        
        overlap_v = max(0,y1- y0)/max(height1,height2)
        return overlap_v

    def sub_graphs_connected(self):
        sub_graphs=[]
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v=index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v=np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs

    def fit_box_2(self,text_boxes):
        '''
        先用所有text_boxes的最大外包点做，后期可以用线拟合试试
        '''
        x1 = np.min(text_boxes[:,0,0])
        y1 = np.min(text_boxes[:,0,1])
        x2 = np.max(text_boxes[:,2,0])
        y2 = np.max(text_boxes[:,2,1])
        return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]

    def fit_box(self,text_boxes):
        x1 = np.min(text_boxes[:,0,0])
        y1 = np.mean(text_boxes[:,0,1])
        x2 = np.max(text_boxes[:,2,0])
        y2 = np.mean(text_boxes[:,2,1])
        return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]

    def get_text_line(self):
        #在textline对时候某些iou不够对框会跳过去，这个和独立对框是不同对
        #这里申请一个临时存贮对list在保存这些跳过去对框
        for idx ,_ in enumerate(self.rects):
            sucession = self.get_sucession(idx)
            if(sucession>0 and sucession not in self.tmp_connected ):
                # and sucession not in self.tmp_connected
                self.tmp_connected.append(sucession)
                self.graph[idx][sucession] = 1 
           

        text_boxes = []
        sub_graphs = self.sub_graphs_connected()
        for sub_graph in sub_graphs:
            tb = self.rects[list(sub_graph)]
            tb = self.fit_box_2(tb)
            text_boxes.append(tb)
        text_boxes = np.array(text_boxes)


        #独立未合并的框 , 和 iou不够 没有参与textline的框
        set_element = set([y for x in sub_graphs for y in x])
        for idx,_ in enumerate(self.rects):
            if(idx not in set_element):
                inner_box = self.rects[[idx]]
                inner_box = self.fit_box_2(inner_box)
                if self.isInside(inner_box,text_boxes) == False:
                    inner_box = np.expand_dims(np.array(inner_box),axis=0)
                    if(text_boxes.shape[0] ==0):
                        text_boxes = inner_box
                    else:
                        text_boxes = np.concatenate((text_boxes,inner_box),axis=0)
                    sub_graphs.append([idx])
        # inv offset
        text_boxes = text_boxes - self.offset
        return text_boxes , sub_graphs


    def get_text_line_morpholopy_closing(self):
        text_boxes,sub_graphs = self.get_text_line()
        self.morphology_closing_combine(text_boxes)


    def get_text_line_split_cell(self,cell_lines):
        text_boxes,sub_graphs = self.get_text_line()
        text_boxes = self.split_by_cell(text_boxes,sub_graphs,cell_lines)
        return text_boxes


    def isInside(self,inner_rect,out_rects):
        """
            判断相交面积大于等于自身面积则为在内部
        """
        if(out_rects.shape[0] ==0):
            return False
        inner_area = abs(inner_rect[0][0] - inner_rect[1][0]) * abs(inner_rect[0][1] - inner_rect[3][1])
        
        x1 = np.maximum(inner_rect[0][0],out_rects[:,0,0])
        y1 = np.maximum(inner_rect[0][1],out_rects[:,0,1])
        x2 = np.minimum(inner_rect[2][0],out_rects[:,2,0])
        y2 = np.minimum(inner_rect[2][1],out_rects[:,2,1])

        w = np.maximum(0.0,x2 - x1)
        h = np.maximum(0.0,y2 - y1)
        interstion_area = w  * h 
        if(((interstion_area / inner_area)>0.95)).any():
            return True
        else:
            return False


    def split_by_cell(self,text_boxes,sub_graphs,cell_lines):
        '''
        根据表格线切分textline。
        get_text_line 得到 text_boxes ,sub_graphs对应每个text_box 由哪些框rect组成
        遍历每个text_box,如果被表格线截断：
        1.切断text_box 但是未切断rect
        2.切断text_box 也切断rect
        '''
        split_boxes = [] 
        #cell_line按x坐标排序
        cell_lines = sorted(cell_lines,key = lambda k:k[0])
        cell_lines = np.array(cell_lines)
        for text_box , sub_grah in zip(text_boxes , sub_graphs):
            
            res = [self.isSplit(cell_line , text_box) for cell_line in cell_lines]
            if(np.array(res).any() == False):  #没有切断text_box
                split_boxes.append(text_box)
                continue
                    
            ###切断了text_box###
            sub_rects = self.rects[sub_grah]
            #小框按x坐标排序
            sub_rects = sorted(sub_rects , key = lambda k:k[0][0],reverse = True )
            s_cell_lines = cell_lines[res]
            for s_c_l in s_cell_lines:  #遍历cell_line 
                tmp_rects = [] 
                s_c_l_x = s_c_l[0]
                while(len(sub_rects)>0):
                    tmp_rect = sub_rects.pop()
                    left_x , right_x= tmp_rect[0][0] , tmp_rect[1][0]
                    #print('cell line , left_x {} right_x {} x {}'.format(left_x,right_x,s_c_l_x))
                    if(s_c_l_x >= right_x): ##cell_line 在小框右侧
                        tmp_rects.append(tmp_rect)
                    elif(s_c_l_x < right_x  and s_c_l_x > left_x): ## cell_line在小框中间
                        left_rect = tmp_rect.copy()
                        left_rect[1][0] = s_c_l_x
                        left_rect[2][0] = s_c_l_x
                        right_rect = tmp_rect.copy()
                        right_rect[0][0] = s_c_l_x
                        right_rect[3][0] = s_c_l_x
                        tmp_rects.append(left_rect)
                        sub_rects.append(right_rect)
                        break
                    else:       ## cell_line 在小框左侧
                        sub_rects.append(tmp_rect)
                        break
                if(len(tmp_rects)>0):
                    split_boxes.append(self.fit_box_2(np.array(tmp_rects)))
            if(len(sub_rects)>0):
                split_boxes.append(self.fit_box_2(np.array(sub_rects)))
        return split_boxes


    def isSplit(self,cell_line,box):
        """
        判断线段和矩形相交
        这里采用简单方法，假设cell_line为垂直的
        x1 在 bx1 bx2之间
        y1,y2 与by1 by2 有重合的地方
        """
        x1,y1,_,y2 = cell_line
        bx1,by1,bx2,by2 = box[0][0],box[0][1],box[2][0],box[2][1]

        if(x1 > bx1 and x1 < bx2):
            h = by2 - by1
            y1 = max(y1,by1)
            y2 = min(y2,by2)
            overlap_v = max(0,y2- y1)/h
            if(overlap_v > 0.5):
                return True
        return False

def morphology_closing_combine(rects):
    '''
    使用闭运算做联通域，对文字块内对rects采用大值text_line
    '''
    g = text_porposcal(rects,max_dist=20,scale_h = 1.5,threshold_overlap_v=0.5)
    rects,_ = g.get_text_line()
    if(len(rects)<2):
        return rects
    #画一个二值图
    xmax = np.max(rects[:,:,0])
    ymax = np.max(rects[:,:,1])
    xmax = xmax + 10 
    ymax = ymax + 10 
    bin_img = np.zeros((ymax,xmax),dtype = np.uint8)
    for rect in rects:
        cv2.drawContours(bin_img,[rect],-1,(255),-1)
    #cv2.imwrite('source.jpg',bin_img)
    #做一个closing
    cluster_h = rects[:,2,1] - rects[:,0,1]
    cluster_h = np.reshape(cluster_h/2,(-1,1))
    km = KMeans(n_clusters=1).fit(cluster_h)
    c_h = km.cluster_centers_[0][0]
    #print('c_h:',c_h)
    kernel = np.ones((int(c_h),2))
    closed_img = cv2.morphologyEx(bin_img,cv2.MORPH_CLOSE,kernel)
    #cv2.imwrite('closing.jpg',closed_img)
    _,cnts,_ = cv2.findContours(closed_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    combine_rects = [[] for i in range(len(cnts))]
    for rect in rects:
        for idx , cnt in enumerate(cnts):
            center_x , center_y = (rect[0][0] + rect[2][0]) //2 ,(rect[0][1] + rect[2][1]) //2
            inside = cv2.pointPolygonTest(cnt,(center_x,center_y),False)
            #print('inside:',inside)
            if(inside == 1):
                #print('rect',rect)
                combine_rects[idx].append(rect)
    
    text_line_rects = [] 
    print(len(combine_rects))
    for c_rts in combine_rects:
        if(c_rts == []):
            continue
        g = text_porposcal(c_rts,max_dist=100,scale_h = 5,threshold_overlap_v=0.5)
        rts,_ = g.get_text_line()
        if(text_line_rects ==[]):
            text_line_rects = rts
        else:
            text_line_rects = np.vstack((text_line_rects,rts))
    return text_line_rects
