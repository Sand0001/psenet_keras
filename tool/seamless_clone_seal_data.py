import random
import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
class SeamClone():
    def __init__(self):
        self.pic_path = '/fengjing/test_img/eng_tmp/image'
        self.seal_path = '/data2/fengjing/pse_data/fg'
        self.rotate_or_not = True
        self.process_num = 0

    def prob(self,c):
        return True if c*10 < random.randint(1,11) else False

    def apply_seamless_cloe_add_foreground(self, img1):
        '''

        :param img1:
        :return:
        '''
        #tmp_name = '/data1/fengjing/output/tmp/' + str(time.time() + random.random()) + '.jpg'
        tmp_name = '/fengjing/data_script/OCR_textrender/output/tmp/' +'1.jpg'
        cv2.imwrite(tmp_name, img1)
        img1 = cv2.imread(tmp_name)
        img2 = random.choice(self.fgs)
        height, width = img1.shape[0:2]
        height_2, width_2 = img2.shape[0:2]
        # 最大crop img 的宽高
        crop_max_width = min(width, width_2)
        crop_max_height = min(height_2, height)
        # 实际crop img 宽高
        crop_height = random.randint(5, crop_max_height)
        crop_width = random.randint(5, crop_max_width)
        # crop img 随机裁剪的位置
        crop_x = random.randint(0, crop_max_width - crop_width)
        crop_y = random.randint(0, crop_max_height - crop_height)
        crop_img = img2[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        crop_h, crop_w = crop_img.shape[0:2]
        # 随机x，y 框放的位置
        range_x = random.randint(0, width - crop_w)
        range_y = random.randint(0, height - crop_h)

        # 由随机xy 计算center
        center = (range_x + crop_w // 2, range_y + crop_h // 2)
        mask = 255 * np.ones(crop_img.shape, crop_img.dtype)
        mixed_clone = cv2.seamlessClone(crop_img, img1, mask, (center[0], center[1]), cv2.MIXED_CLONE)
        ret, binary = cv2.threshold(~crop_img[:, :, 2], 30, 255, cv2.THRESH_BINARY)   #二值化求边缘
        mask_coor = np.argwhere(binary > 200)

        for i in mask_coor:
            try:
                img1[range_y + i[0], range_x + i[1]] = mixed_clone[range_y + i[0], range_x + i[1]]
            except Exception as e:
                print(e)
                continue

        np_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        return np_img

    def rotate(self,img,angle,mask = False):

        rows, cols = img.shape[:2]
        center = (cols / 2, rows / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        if mask == True:
            borderValue = 0
        else:
            borderValue = (255,255,255) if len(img.shape) == 3 else 255
        dst = cv2.warpAffine(img, M, (cols, rows),borderValue = borderValue )

        return dst

    def dilate(self,binary):
        k = random.randint(2,4)
        # 设置卷积核
        kernel = np.ones((k, k), np.uint8)
        # 图像膨胀处理
        binary = cv2.dilate(binary, kernel)
        return binary

    def semless_clone(self,picInfo,pic_seal_num,all_pic_num):
        seal_list = os.listdir(self.seal_path)

        for key in picInfo:
            picList = picInfo[key]
            pic_seal_list = []

            random.shuffle(picList)
            print('int(pic_seal_num*1.0/all_pic_num*len(picList))',int(pic_seal_num*1.0/all_pic_num*len(picList)))
            print('len(pic_seal_list)',len(pic_seal_list))
            # for i in range(int(pic_seal_num*1.0//all_pic_num*len(picList))):
            while (len(pic_seal_list)<int(pic_seal_num*1.0/all_pic_num*len(picList))):
                pic = random.choice(picList)


                if pic not in pic_seal_list:
                    #print('i',i)

                #for pic in picList:
                    if 'jpg'  in pic or 'png' in pic:
                        img = cv2.imread(os.path.join(key,pic))
                        height, width = img.shape[0:2]

                        seal_pic = random.choice(seal_list)
                        try:
                            random_style = random.randint(0,5)
                            if random_style ==1 :
                                seal_img = cv2.imread(os.path.join(self.seal_path,seal_pic))[:,:,[2,1,0]]

                            elif random_style == 2:
                                seal_img = cv2.imread(os.path.join(self.seal_path,seal_pic),cv2.IMREAD_GRAYSCALE)
                                seal_img = cv2.cvtColor(seal_img,cv2.COLOR_GRAY2BGR)
                            else:
                                seal_img = cv2.imread(os.path.join(self.seal_path, seal_pic))

                            #binary = self.rotate(binary)
                            #ret, binary = cv2.threshold(~seal_img[:, :, 2], 30, 255, cv2.THRESH_BINARY)  # 二值化求边缘

                            mask = 255 * np.ones(seal_img.shape, seal_img.dtype)
                            if self.prob(0.5):

                                angle = random.randint(0,360)
                                print('rotate',angle)
                                mask = self.rotate(mask,angle,mask = True)
                                seal_img = self.rotate(seal_img,angle)
                                #binary = self.dilate(binary)
                            height_seal ,width_seal = seal_img.shape[:2]

                            # 求图片粘贴的位置
                            # 随机x，y 框放的位置
                            range_x = random.randint(0, width - width_seal)
                            range_y = random.randint(0, height - height_seal)

                            # 由随机xy 计算center
                            center = (range_x + width_seal // 2, range_y + height_seal // 2)
                            mixed_clone = cv2.seamlessClone(seal_img, img, mask, (center[0], center[1]), cv2.MIXED_CLONE)
                            ret, binary = cv2.threshold(~seal_img[:, :, 2], 30, 255, cv2.THRESH_BINARY)  # 二值化求边缘

                            mask_coor = np.argwhere(binary > 200)
                            for i in mask_coor:
                                try:
                                    img[range_y + i[0], range_x + i[1]] = mixed_clone[range_y + i[0], range_x + i[1]]

                                except Exception as e:
                                    print(e)
                                    continue
                            cv2.imwrite(os.path.join(key,pic),img)
                            self.process_num += 1
                            print('已经处理完成图片 {} 张'.format(self.process_num))

                        except:
                            continue
                    pic_seal_list.append(pic)


if __name__ == '__main__':
    import sys
    SeamClone = SeamClone()
    data_path = sys.argv[1]
    #data_path = '/fengjing/test_img/eng_tmp'
    num = 0
    pic_seal_num = 110
    picList_all = []
    pic_info = {}
    for path in os.listdir(data_path):
        if 'text_zixuan' in path:
            if data_path[-1] != '/':
                data_path = data_path + '/'
            pic_path = data_path+path+'/image'
            # pic_path = data_path+path
            picList = os.listdir(pic_path)
            pic_info[pic_path] = picList
            num += len(picList)
    print('num',num)
    SeamClone.semless_clone(pic_info,pic_seal_num,num)



