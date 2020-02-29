
## keras implementation of PSENET 
For more detials , please see the the original paper : [Shape Robust Text Detection with Progressive Scale Expansion Network](https://arxiv.org/abs/1806.02559)

百度网盘:链接：https://pan.baidu.com/s/1U1dYc4sYxKtoqSGMgP_n4A 提取码：40dh 

### Results:
<center class='half'>
    <img src="imgs/res1.png" width='400'> <img src="imgs/res2.png" width='400'>
    <img src="imgs/res3.png" width='400'> <img src="imgs/res4.png" width='400'>
    <img src="imgs/sfz.png" width='400'> <img src="imgs/jsz.jpg" width='400'>
</center>

### PSENET for ICPR MTWI 2018 Challenge 2 Text detection.
| Method | Precision (%) | Recall (%) | F-measure (%) | 
| - | - | - | - |
| PSENet-2s-resnet50 | 72.2 | 68.7 | 0.704 |


conda acitvate env 激活python3环境，如果不激活python3环境会出现，找不到python.h文件。
编译 cpp文件
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` pse.cpp -o pse`python3-config --extension-suffix`

