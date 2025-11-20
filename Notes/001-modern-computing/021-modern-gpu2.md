---
title: 021. 现代显卡之二：T&L、RCP与GeForce 256的历史演进
tags:
  - GPU
  - 图形处理器
  - RCP
  - 硬件历史
  - 3D渲染
---
# 021. 现代显卡之二

## T&L

上一节曾说过，GPU的主要作用是把3D场景变成一个2D图像。在没有GPU的时候，这部分工作要么由照像机来做，要么由人眼来做。这里涉及的主机是几何学。
主要需要解决的问题为Transform, clipping and lignting, 简称T&L或TCL。

1. Transform

即将一个3D物体变化为2D图像，基本做法是投影，包括透视投影和垂直投影。
其实这里面还会涉及把物体的“物体坐标”转换成世界坐标，再转换成“相机坐标”等。这涉及一系列矩阵变换。

2.  clipping

3D物体投影到2D平面上后，或是投影过程中，发现有一些部分是不应该被看见的，这时需要根据遮挡信息等把他们裁掉。

3.  lighting

光把3D物理投影到2D平面并不能给人们带来足够的真时感，甚至会投影成混乱一团像素，让人看不懂倒底是什么东西。因为投影的过程中缺少光照信息。同样的几何体，由于光照角度不同，材质不同，可能会显示出非常不一样的颜色，人眼需要这种颜色或灰度变化来区分物体间的位置关系。

在显卡被发明之前，一些做游戏的公司就已经发现了这套理论，1993年arcade公司就实现了能做T&L的硬件，1997年任天堂的游戏机上也有了专门的T&L硬件。PC上由于人们认为PC的CPU性能足够强大，所以很长一段时间内，大家都用软件去实现T&L。直到1999年，nvidia的GeForce 256才引入硬件T&L支持。

本小节我们就来看看这段历史。

## RCP

RCP，即Reality Coprocessor， 是由Silicon Graphics公司为任天堂研发的。它包含两部分，RSP和RDP，即Reality Signal Processor和Reality Display Processor。

### RSP
RSP的主要做是几何运算，对应的是T&L中的T和L。
RDP的主要作用是栅格化和纹理贴图。

![](/assets/f11-01.gif)

RSP是一颗MIPS CPU，它的固件实现了一个程序，这个程序可以读取一个指令序列，解析指令序列中的指令并执行相应的指令。这些指令称为微码，micro code。程序员将一系列微码组成一个列表，称为display list或是command list。RSP则按照display list去执行。

![](/assets/f11-01-00.gif)

RSP的微码看起来像是C函数调用，但实际上它们可能是一些打包二进制比特流的宏。熟悉opengl的同学可能会觉得它们和opengl的API有点像。

其操作思路与opengl也是很相似的，先要指定一些顶点，一些变换矩阵，一些纹理，对应的api如下：
```c

gsSPMatrix(Mtx *m, unsigned int p) 
gsSPVertex(Vtx *v, unsigned int n, unsigned int v0) 
gsSPTexture(int s, int t, int levels, int tile, int on) 
gsSPClipRatio(FRUSTRATIO_3),
Lights3 light_structure1 = gdSPDefLights3(
        ambient_red, ambient_green, ambient_blue,
        light1red, light1green, light1blue,   
                   light1x, light1y, light1z,
        light2red, light2green, light2blue,   
                   light2x, light2y, light2z,
        light3red, light3green, light3blue,   
                   light3x, light3y, light3z);
gsSPSetGeometryMode(G_LIGHTING),
gsSPSetLights3(material1_light),
gsSPVertex( /* define vertices for object 1 */ );
/* render object 1 here */
gsSPLight(&material2_light.l[0], LIGHT_1),
gsSPLight(&material2_light.a, LIGHT_2),
gsSPVertex( /* define vertices for object 2 */ );
/* render object 2 here */
guLookAtHilite(&throw_away_matrix, &lookat, &hilite,
                Eyex,      Eyey,    Eyez,
                Objectx,   Objecty, Objectz,
                Upx,       Upy,     Upz,
                light1x,   light1y, light1z,
                light2x,   light2y, light2z,
                tex_width, tex_height);
gsDPLoadTextureBlock_4b(hilight_texture, G_IM_FMT_I,
                        tex_width, tex_height, 0,
                        G_TX_WRAP | G_TX_NOMIRROR,
                        G_TX_WRAP | G_TX_NOMIRROR,
                        tex_width_power2,
                        tex_height_power2,
                        G_TX_NOLOD, G_TX_NOLOD),

```

具体的含义见：
http://level42.ca/projects/ultra64/Documentation/man/pro-man/pro11/index11.6.html

RCP的输入是一些顶点、纹理坐标、光照条件等，输出是多边形。
这些多边形进入RDP进行进一步处理。

### RDP
Reality Display Processor。
它的作用是把多边形进行栅格化，输入为```high-quality, Silicon Graphics style pixels that are textured, antialiased, and z-buffered.```

听起来的大致意思是这是一些像素，每个像素都已经映射了纹理，做了抗锯齿，并且包含z缓冲即深度信息。与opengl里面的fragment,片元几乎是一回事儿。

RDP会按照固定的步骤处理这些片元。在某种模式下的步骤如下：

![](/assets/f12-01.gif)

在这种模式下，它一个周期只能处理1个像素，在另一些模式下，它可以实现一个周期处理两个像素。

RCP已经很像是一款固定流水线的GPU了，美中不足的是它的RSP部分是用MIPS处理器来做的，速度就慢了不少。


|Block| Functionality|
|-|-|
|RS|Generates pixel and its attribute covered by the interior of the primitive.|
|TX	|Generates 4 texels nearest to this pixel in a texture map.|
|TF	|Bilinear filters 4 texels into 1 texel,or performs step 1 of YUV-to-RGB conversion.|
| CC	| Combines various colors into a single color,or performs step 2 of YUV-to-RGB conversion.|
|BL	|Blends the pixel with framebuffer memory pixel, or fogs the pixel for writing to framebuffer.|
|MI	|Fetches and writes pixels from and to the framebuffer memory.|


## GeForce 256

![](/assets/440px-KL_NVIDIA_Geforce_256.jpg)

这个GPU支持Direct3D 7。称为世界上的第一个GPU。

Nvidia如些骄傲于这个芯片，已至于官网上还有它的链接：
https://www.nvidia.com/page/geforce256.html


这时GPU的定义为：```a single-chip processor with integrated transform, lighting, triangle setup/clipping, and rendering engines that is capable of processing a minimum of 10 million polygons per second.```

一秒钟可以得处理1000万个多边形。

256的意思是”256位，4流水线渲染引擎“。相当于4核的GPU，一个周期可以处理4个像素。

与RCP不同的是，所有的处理都是硬逻辑完成的，而不是RCP那像由运行于CPU上的固件完成。

Nvidia不愧为世界上第一个GPU，它有现代GPU的所有特点：
1. 硬件T&L
2. 多核
3. 支持一款3D API。





可以说硬件支持T&L是GPU和非GPU的分界点，Direct3D 7，是固定流水线GPU和非固定流水线分界点。当Direct3D 8出来的时候，vertex shader, fragment shader就已经有了，这时就是非固定流水线的GPU了。

太晚了，今天就写到这里。

发现一个好文章还没读完，供大家参考：
https://www.techspot.com/article/653-history-of-the-gpu-part-2/






