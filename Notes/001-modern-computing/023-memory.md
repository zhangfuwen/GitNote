# memory

![](assets/Pasted%20image%2020251024163451.png)

## DDR

| 内存分类            | 	传输带宽      |
|-----------------|------------|
| DDR	"DDR 266    | 2.1 GB/s   |
| DDR 333         | 2.6 GB/s   |
| DDR 400         | 3.2 GB/s"  |
| DDR2 533  | 4.2 GB/s   |
| DDR2 667        | 5.3 GB/s   |
| DDR2 800        | 6.4 GB/s"  |
| DDR3 1066 | 8.5 GB/s   |
| DDR3 1333       | 10.6 GB/s  |
| DDR3 1600       | 12.8 GB/s  |
| DDR3 1866       | 14.9 GB/s" |
| DDR4 2133 | 17 GB/s    |
| DDR4 2400       | 19.2 GB/s  |
| DDR4 2666       | 21.3 GB/s  |
| DDR4 3200       | 25.6 GB/s" |
| DDR5 4800 | 38.4 GB/s  |
| DDR5 5200       | 41.6 GB/s  |
| DDR5 5400       | 43.2 GB/s  |
| DDR5 5600       | 44.8 GB/s  |
| DDR5 6000       | 48.0 GB/s  |
| DDR5 6400       | 51.2 GB/s" |


![img.png](assets/img.png)

![img.png](assets/img1.png)

## LPDDR

面向AI的手机面临内存带宽瓶颈问题，所以一般采用最新的LPDDR技术。

基于高通8gen3平台或MTK 9300平台的SoC，一般采用LPDDR5X,即4800MHz的LPDDR5内存，标称内存带宽是76.8GB/s，实际可以达到60GB/s。
基于高通8gen4平台或MTK 9400平台的SoC, 一般采用LPDDR5X plus，即5333MHz的LPDDR5内存，标称内存带宽是85.3GB/s，实际可以达到xxGB/s。

![soc memory](assets/soc-memory1.png)

LPDDR5标准中，一个通道是16根（差分对，下同）线，一般手机用64根线，即4通道。单根线的数据率是9600Mbps(对应4800MHz)，总带宽是9.6GT/s * 8 bytes/T = 76.8GB/s。

LPDDR6单根线可达14.4Gbps, 对应频率应该是7200MHz。但是比较不同的是，LPDDR6一个通过是24根线的，这个数儿无法被64整除。合理的猜测是手机会用2通道(48根)或3通道（72根）。

这样总的带宽应该是14.4GT/s * 6 bytes/T = 86.4GB/s或 14.4GT/s * 9 bytes/T = 129.6 GB/s。如果是后者的话，带宽是LPDDR5X的1.69倍，是LPDDR5X plus的1.51倍。

[参数资料](https://www.jedec.org/sites/default/files/Brett%20Murdock_FINAL_Mobile_2024.pdf)

### 三星

在三星官网上查LPDDR5X的芯片，看到的结果如下：

![](assets/Pasted%20image%2020251024171641.png)
解读了下这里的数据：
第一列是型号
第二列是density，就是一片有多大容量，64Gb就是8GB。32Gb就是4GB，就比较少。
第三列是封装形式。
第四列是最大的数据率，即数据引脚上的数据输出速度。这个数乘以第七列的位宽就是理论最大带宽。9600Mbps + x64就是76.8GB/s。
第五列是电压，LPDDR5X有输入电压，依次是VDD1, VDD2H, VDD2L和VDDQ。
第六列是温度范围。
第七列是量产还是样片，这里我筛选了量产的。

### 美光

从网上找到一个镁光的lpddr5x的芯片手册：
[Micron芯片手册](assets/Micron_05092023_315b_441b_y4bm_ddp_qdp_8dp_non_aut-3175604.pdf)
面里看到的额外信息：
![](assets/Pasted%20image%2020251024173209.png)

位宽：x32位宽的版本实际上是2 channel x 16，x64也是4 channel x 16。
Rank数：里面有单rank和2 rank的，两rank可以认为是一个通道里接了两颗芯片，通过片选来区别，这样的好处时总bank数多了，实际效果上的page变大了，会对时延有帮助。

![](assets/Pasted%20image%2020251024173535.png)
可以看到并是多少channel就多少个die，也可能一个channel是两个die组在一起的。

![](assets/Pasted%20image%2020251024173858.png)
bank/bank group: 有些芯片是8 bank, 16 bank，但是没有分成多个bank group的。
Page size: 差别比较大，有1K的也有4K的。
这个手册里没提及CL等参数，所以对访存时延很难判断。问千问，千问说LPDDR5X 9600MT/s的CL一般为48~64，单位是tCK，9600MT/s时，tCK是1/4800M秒，约0.2083ns，所以CL时延约为10~13ns。它还给了个表：
![](assets/Pasted%20image%2020251024175124.png)
从表上看，CL按ns算的值其实没太大变化，都是10~15ns左右。这跟知乎上一个同学说的是一致的，即CL从时延上几乎没有变，频率快了，CL的值就变大了。

### 电源与功耗

LPDDR5X有四个power rail。VDDQ是外围电路。VDD2L也是外围电路。VDD2H和VDD1主要是在bank上。

![](assets/Pasted%20image%2020251024175542.png)
现有方案中，VDD2L和VDD2H的分离已经能带来一些功耗下降。镁光提了一个更低VDD2H的方案，能进一步降低 
## UFS

UFS 4的理论带宽是4GB/s，它有2个lane(差分线对)，单个差分线对速度是23.2GT/s（不算头部开销理论传输速度是5.8GB/s）.

现在市面上的手机一般采用UFS 4.0。存储访问带宽与介质有关，与容量（内部bank个数）也相关。目前已知最快的实际速度为3.8GB/s。

UFS 4.1理论上可以达到8GB/s，本质上就是4 lane的UFS 4。

UFS 的jedec网址是：https://www.jedec.org/sites/default/files/docs/JESD220.pdf


