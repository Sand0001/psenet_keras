#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np 

def cal_angle(thin_img,threshold):
    lines = cv2.HoughLines(thin_img,1,np.pi/360,threshold)
    if(lines is None):
        return None
    angles = []
    for line in lines:
        rho,theta = line[0]
        ## 精度0.5
        angles.append(theta * 180 / np.pi  // 0.2 * 0.2 )
    return Counter(angles).most_common(1)[0][0]

img = cv2.imread('./tmp/test.jpg')
img_rgb = np.copy(img)
#%%

#%%
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(img.shape)
#img = cv2.threshold(img,125,255,cv2.THRESH_BINARY)


#%%
_,img= cv2.threshold(img,125,255,cv2.THRESH_BINARY_INV)

#%%

plt.imshow(img,cmap='gray')


#%%
import numpy as np 
thin_img = img.astype(np.uint8)
thin_img = cv2.ximgproc.thinning(thin_img)
#%%
cv2.imwrite('thi.jpg',thin_img)
thin_img.dtype
#%%
lines = cv2.HoughLines(thin_img,2,np.pi/360,thin_img.shape[1]//5)
for line in lines:
    rho,theta = line[0]
    print('rho',rho,'thea ',theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 10000 * -b)
    y1 = int(y0 + 10000 * a)
    x2 = int(x0 - 10000 * -b)
    y2 = int(y0 - 10000 * a)
    res = cv2.line(img_rgb,(x1,y1),(x2,y2),(0,255,0),2)
    res = cv2.circle(img_rgb,(x0,y0),10,(255,0,0),3)


#%%
cv2.imwrite('l.jpg',img_rgb)

#%%
plt.imshow(img_rgb)

#%%

len(lines)
#%%


#%%
