# 使用python调用opencv

## 1. 环境搭建

### 1.1 MacOS上面安装

#### 1.1.1 安装OpenCV

使用Homebrew安装直接安装OpenCV:

```bash
brew install opencv
```

OpenCV的安装目录为：/usr/local/Cellar/opencv/。  
也可以从官网下载安装包直接安装。比较复杂的是下载源代码使用cmake安装。

#### 1.1.2 安装cv2

OpenCV在Python中调用的时候使用的库是cv2。在python中可以直接使用cv2:

```python
import cv2
```

直接import会出现错误：

```

  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  ImportError: No module named cv2

```

这是因为Python并不能找到对应的OpenCV的库。  
那cv2在那里呢?使用pip install安装，也没有找到对应的cv2库。回头看OpenCV的安装目录在lib下面可以看到一个python2.7的包，Python的相关引用都在里面，在site-packages目录下有两个文件:

```
cv.py
cv2.so
```

可以把这两个文件复制Python库目录/usr/local/lib/python2.7/site-packages下面，这样在调用的时候Python就可以找到cv2的库。

### 1.2 Ubuntu上安装OpenCV的方法

#### 1.2.1 apt-get安装


```
sudo apt-get install libopencv-dev python-opencv
```


#### 1.2.2 源代码安装

先Ubuntu上可以直接编译OpenCV安装,首先要安装编译需要的依赖包：

```bash
sudo apt-get install cmake build-essential libgtk2.0-dev libjpeg8-dev libjpeg-dev libavcodec-dev libavformat-dev libtiff5-dev cmake libswscale-dev
```

下载OpenCV的源代码编译并安装:

```
wget https://codeload.github.com/o...
tar -xzvf 2.4.13.tar.gz
cd opencv-2.4.13
cmake
make
sudo make install
```

另外在Ubuntu上使用Python调用OpenCV需要安装对应的python包:

```
sudo apt-get install python-opencv
```

### 1.3 Windows环境

#### 1.3.1 OpenCV

1）下载适用于 windows 的 opencv 安装包。OpenCV下载页面：[http://sourceforge.net/projects/opencvlibrary/files/opencv-win/](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/)  
2）直接下载 OpenCV-2.4.5.exe \(278.9 MB\)  
安装。你会发现实际上就是解压缩的过程。

#### 1.3.2 Python

1）下载适用于 windows 的Python安装包。Python下载页面：[http://www.python.org/getit/](http://www.python.org/getit/)  
直接下载 Python 2.7.5 Windows Installer   
2）安装。  
3）扩展Python的第三方库——OpenCV的python库  
进入OpenCV的安装目录下找到：\build\python\2.7\cv2.pyd  
将cv2.pyd复制到Python的子目录下：\Lib\site-packages\

OK，至此环境就搭建好了。

#### 1.3.3 测试示例程序

OpenCV自带了很多示例程序，进入OpenCV安装目录下的子目录：\samples\python2\ ，可以看到很多以.py为后缀名的文件；  
用python随便打开一个.py文件，按F5键运行，看看效果

### 1.4 安装numpy

```shell
sudo apt-get install python-numpy
```

## 2. 使用OpenCV

### 2.1 版本验证

一个简单用来验证Python是否能够调用OpenCV的方法:

```python
import cv2cv2.__version__
```

可以得到OpenCV版本：

'2.4.13.1'

### 2.2 简单加载图片显示



```python
import cv2

img = cv2.imread("./images/1.jpeg")
cv2.namedWindow("ImageWindows")
cv2.imshow("ImageWindows",img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
```
### 2.3 相似度比较
以下代码来自：http://blog.csdn.net/sunny2038/article/details/9057415

```python
# -*- coding: utf-8 -*-
#feimengjuan
# 利用python实现多种方法来实现图像识别
import cv2
import numpy as np
from matplotlib import pyplot as plt
# 最简单的以灰度直方图作为相似比较的实现
def classify_gray_hist(image1,image2,size = (256,256)):
    # 先计算直方图
    # 几个参数必须用方括号括起来
    # 这里直接用灰度图计算直方图，所以是使用第一个通道，
    # 也可以进行通道分离后，得到多个通道的直方图
    # bins 取为16
    image1 = cv2.resize(image1,size)
    image2 = cv2.resize(image2,size)
    hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0])
    hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0])
    # 可以比较下直方图
    plt.plot(range(256),hist1,'r')
    plt.plot(range(256),hist2,'b')
    plt.show()
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i]))
        else:
            degree = degree + 1
    degree = degree/len(hist1)
    return degree
# 计算单通道的直方图的相似值
def calculate(image1,image2):
    hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0])
    hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0])
     # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i]))
        else:
            degree = degree + 1
    degree = degree/len(hist1)
    return degree
# 通过得到每个通道的直方图来计算相似度
def classify_hist_with_split(image1,image2,size = (256,256)):
    # 将图像resize后，分离为三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1,size)
    image2 = cv2.resize(image2,size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1,im2 in zip(sub_image1,sub_image2):
        sub_data += calculate(im1,im2)
    sub_data = sub_data/3
    return sub_data
# 平均哈希算法计算
def classify_aHash(image1,image2):
    image1 = cv2.resize(image1,(8,8))
    image2 = cv2.resize(image2,(8,8))
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    hash1 = getHash(gray1)
    hash2 = getHash(gray2)
    return Hamming_distance(hash1,hash2)
def classify_pHash(image1,image2):
    image1 = cv2.resize(image1,(32,32))
    image2 = cv2.resize(image2,(32,32))
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct1 = cv2.dct(np.float32(gray1))
    dct2 = cv2.dct(np.float32(gray2))
    # 取左上角的8*8，这些代表图片的最低频率
    # 这个操作等价于c++中利用opencv实现的掩码操作
    # 在python中进行掩码操作，可以直接这样取出图像矩阵的某一部分
    dct1_roi = dct1[0:8,0:8]
    dct2_roi = dct2[0:8,0:8]
    hash1 = getHash(dct1_roi)
    hash2 = getHash(dct2_roi)
    return Hamming_distance(hash1,hash2)
# 输入灰度图，返回hash
def getHash(image):
    avreage = np.mean(image)
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash

# 计算汉明距离
def Hamming_distance(hash1,hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num

if __name__ == '__main__':
    img1 = cv2.imread('images/1.jpeg')
    cv2.imshow('img1',img1)
    img2 = cv2.imread('images/2.jpeg')
    cv2.imshow('img2',img2)
    degree = classify_gray_hist(img1,img2)
    #degree = classify_hist_with_split(img1,img2)
    #degree = classify_aHash(img1,img2)
    #degree = classify_pHash(img1,img2)
    print degree
    cv2.waitKey(0)
```



