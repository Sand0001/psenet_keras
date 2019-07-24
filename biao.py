#coding =utf-8
'''
'''
import os 
import glob
import tqdm
import cv2
import requests
import json
import csv 

url = r'http://39.104.88.168/ocr_pse_test?language=CHE'
dir = r'/Users/mahuichao/Documents/PSENET/test/1'
img_files = glob.glob(os.path.join(dir,'*.jpg'))


for img_file in tqdm.tqdm(img_files):
    files = {'file':('1.jpg',open(img_file,'rb'),'image/png')}
    res = requests.post(url,files = files)
    js = json.loads(res.text)
    js_name = os.path.splitext(img_file)[0] + '.json'
    with open(js_name,mode = 'w' , encoding = 'utf-8') as f :
        json.dump(js,f,ensure_ascii=False)

