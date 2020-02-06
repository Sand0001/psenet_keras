import os
import sys
import cv2
import time
import logging
import numpy as np 
import tensorflow as tf 

# os.environ["CUDA_VISIBLE_DEVICES"] = '7'
sys.path.append(os.getcwd() + '/psenet')

from utils_up4 import scale_expand_kernels ,text_porposcal , fit_boundingRect_cpp ,fit_minarearectange_cpp
from utils_up4 import calc_vote_angle , fit_boundingRect_warp_cpp



def save_mask_pic(images,labelimage,recname):
    images_tmp = np.reshape(images, images.shape[1:])
    ww = images_tmp.shape[1]
    hh = images_tmp.shape[0]
    labelimage = labelimage.astype(np.uint8)
    labelimage_resize = cv2.resize(labelimage, (ww, hh))
    label_img_tmp = np.zeros(images_tmp.shape, np.uint8)

    label_img_tmp[np.where(labelimage_resize > 0)] = (255, 0, 0)
    alpha = 0.7
    beta = 1 - alpha
    gamma = 0
    print(label_img_tmp.shape, images_tmp.shape)
    img_tmp = cv2.addWeighted(images_tmp, alpha, label_img_tmp, beta, gamma)
    cv2.imwrite('/data/fengjing/ocr_recognition_test/html/image_rec/' + recname, img_tmp)


def predict(images,recname,angle = True, combine=False, lines=[]):
    a = time.time()
    MIN_LEN = 32
    MAX_LEN = 2200
    h, w = images.shape[0:2]
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

    w = int(w //32 * 32)
    h = int(h//32 * 32)

    scalex = images.shape[1] / w
    scaley = images.shape[0] / h

    images = cv2.resize(images,(w,h),cv2.INTER_AREA)
    images = np.reshape(images,(1,h,w,3))   

    res = sess.run([op], feed_dict={ip:images})
    b =time.time()
    logging.info('pse的模型预测耗时：{}'.format(b-a))
    res1 = res[0][0]
    res1[res1>0.9]= 1
    res1[res1<=0.9]= 0
    newres1 = []
    for i in range(0,5):
        n = np.logical_and(res1[:,:,5],res1[:,:,i]) * 255
        n = n.astype('int32')
        newres1.append(n)
    # newres1.append((res1[:,:,5]*255).astype('int32'))

    # num_label,labelimage = scale_expand_kernels(newres1,filter=False)
    # print('7'*30)
    # print(type(angle))
    # print(angle)
    lines = [[x1, y1, x2, y2] if y2 > y1 else [x2, y2, x1, y1] for x1, y1, x2, y2 in lines]
    lines =[ [l[0]/scalex/2,l[1]/scaley/2,l[2]/scalex/2,l[3]/scaley/2] for l in lines]
    degree = 0
    if(angle == False):
        # print("*"*30)
        num_label,labelimage = scale_expand_kernels(newres1,filter=False)
        print('recName',recname)
        labelimage_tmp = labelimage.copy()
        labelimage_tmp[labelimage_tmp > 0] = 255
        # cv2.imwrite('/data/fengjing/ocr_recognition_test/html/image_rec/'+recname,labelimage)
        rects = fit_boundingRect_cpp(num_label-1,labelimage)
        save_mask_pic(images, labelimage, recname)
       
        # rects = morphology_closing_combine(rects)
        g = text_porposcal(rects,max_dist=10,threshold_overlap_v=0.5)
        # logging.info(str(lines))
        # logging.info(type(lines))
        rects = g.get_text_line(lines, combine) # 表格检测版的text连接，lines为空时➡️ gennaral 连接
        rects = np.array(rects)
    else:

        #计算角度
        degree = calc_vote_angle(newres1[-1])
        num_label,labelimage = scale_expand_kernels(newres1,filter=False)
        labelimage_tmp = labelimage.copy()
        labelimage_tmp[labelimage_tmp > 0] = 255
        # cv2.imwrite('/data/fengjing/ocr_recognition_test/html/image_rec/' + recname, labelimage)
        save_mask_pic(images, labelimage, recname)
        logging.info('pse的角度是%s' %str(degree))
        h,w = labelimage.shape[0:2]
        M = cv2.getRotationMatrix2D((w/2,h/2),degree,1.0)
        neg_M = cv2.getRotationMatrix2D((w/2,h/2),-degree,1.0)

        rects = fit_boundingRect_warp_cpp(num_label-1,labelimage,M)
        g = text_porposcal(rects,max_dist=10,threshold_overlap_v=0.5)
        rects = g.get_text_line(lines, combine) # general text 连接
        rects = np.array(rects)
        if rects.shape[0]>0:
             # print(rects.shape)
             rects = cv2.transform(np.array(rects),neg_M)
             # rects_a = rects.tolist()
    if rects.shape[0]>0:
        rects = rects.reshape(-1,8)
   

    c = time.time()
    logging.info('pse的连接部分耗时：{}'.format(c-b))
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
        rt[4], rt[6] = rt[6], rt[4]
        rt[5], rt[7] = rt[7], rt[5]
        rt = np.append(rt, degree)
        results.append(rt)
    return results

def draw_boxes(img, boxes):
    for box in boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    return img

output_graph_def = tf.GraphDef()
with open('psenet/psenet.pb','rb') as f :
    output_graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(output_graph_def, name='')

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.18
# sess = tf.Session(config=config)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

ip = sess.graph.get_tensor_by_name("input_1:0")
op = sess.graph.get_tensor_by_name("activation_55/Sigmoid:0")


if __name__ == '__main__':
    images = cv2.imread('/data/ocr_train_ctpn/page1.jpg')
    print(images.shape)
    sess, IP, OP = get_ip_op()
    rects = predict(images, sess, IP, OP)

    draw_img = draw_boxes(images, rects)
    cv2.imwrite('imag.jpg', draw_img)
   
    cv2.namedWindow("image", 0)
    cv2.resizeWindow('image', 800, 900)
    cv2.imshow('image', draw_img)
    cv2.waitKey(0)

