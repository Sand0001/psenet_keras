import sys
import json
import os 
import glob
import config 


def read_dataset(json_file_dir,image_dir):
    '''
    json文件转csv
    args：
        json_file_dir: json文件夹路径
        image_dir: 图片所在路径
    '''
    json_files_path = glob.glob(os.path.join(json_file_dir,'*.json'))
    json_files_path = os.listdir(json_file_dir)
    dataset = {}
    for filepath in json_files_path:
        f = open(os.path.join(json_file_dir,filepath),'r').readlines()[0]
        #with open(filepath) as f:
        js = json.loads(f)
        imgfilename = js['path']
        imgfilename = os.path.join(image_dir,os.path.basename(imgfilename))
        bndboxes = js['bndbox'] 
        boxes = [] 
        for bndbox in bndboxes:
            x1 = int(bndbox['x1'])
            y1 = int(bndbox['y1'])
            x2 = int(bndbox['x2'])
            y2 = int(bndbox['y2'])
            x3 = int(bndbox['x3'])
            y3 = int(bndbox['y3'])
            x4 = int(bndbox['x4'])
            y4 = int(bndbox['y4'])
            boxes.append([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
        dataset[imgfilename] = boxes
    return dataset

           

# def read_txt(file):
#     with open(file,'r',encoding='utf-8') as f :
#         lines = f.read()
#     lines = lines.split('\n')
#     gtbox =[]
#     for line in lines:
#         if(line==''):
#             continue
#         pts = line.split(',')[0:8]
#         #convert str to int 
#         x1 = round(float(pts[0]))
#         y1 = round(float(pts[1]))
#         x2 = round(float(pts[2]))
#         y2 = round(float(pts[3]))
#         x3 = round(float(pts[4]))
#         y3 = round(float(pts[5]))
#         x4 = round(float(pts[6]))
#         y4 = round(float(pts[7]))

#         gtbox.append([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
#     return gtbox

# def read_dataset():
#     files = glob.glob(os.path.join(config.MTWI_2018_TXT_DIR,'*.txt'))
#     dataset={}
#     for file in files:
#         basename = '.'.join(os.path.basename(file).split('.')[:-1])
#         imgname = os.path.join(config.MTWI_2018_IMG_DIR,basename+'.jpg')
#         gtbox = read_txt(file)
#         dataset[imgname] = gtbox
#     return dataset


