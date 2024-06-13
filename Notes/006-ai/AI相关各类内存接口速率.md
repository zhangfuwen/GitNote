# 汇总

| GPU | vRAM | Memory interface width |. Memory Bandwidth ||
| --- | --- | --- | --- | --- |
|P4000|8GB GDDR5|256-bit|243 GB/s||
|P5000|16GB GDDR5X|256-bit|288 GB/s||
|P6000|24GB GDDR5X|384-bit|432 GB/s||
|V100|32GB HBM2|4096-bit|900 GB/s||
|RTX4000|8GB GDDR6|256-bit|416 GB/s||
|RTX5000|16GB GDDR6|256-bit|448 GB/s||
|A4000|16GB GDDR6|256-bit|448 GB/s||
|A5000|24GB GDDR6|384-bit|768 GB/s||
|A6000|48GB GDDR6|384-bit|768 GB/s||
|A100|40GB HBM2|5120-bit|1555 GB/s||

GTK780 ![[assets/Pasted image 20240613160328.png]] 


# DDR


| Generation | 时钟频率              | MT/s(clock rate的二倍) | burst chop | burst length | per chip data width | per channel (64bit) band width |
| ---------- | ----------------- | ------------------- | ---------- | ------------ | ------------------- | ------------------------------ |
| DDR3       | 800MHz ~  1600MHz | 1600MT/s ~ 3200MT/s | 4          | 8            | x4, x8, x16         | 25.6 GB/s                      |
| DDR4       | 1066MHz ~ 2666MHz | 2133MT/s ~ 5333MT/s | 4          | 8            | x4, x8, x16         | 42.6 GB/s                      |
| DDR5       | 2000MHz ~ 4000MHz | 4GT/s ~ 8GT/s       | 8          | 16           | x32+x8 ECC          | 64 GB/s                        |


## DDR3

[jedec spec](assets/JESD79-3E.pdf)

DDR3用于PC和服务器架构。
一个Socket(CPU)可能有多个Memory Controller。一个Controller可以支持多个channel，通常就是dual channel。一个channel代表一个DIMM条。一个DIMM条可以有多个rank，通常是两个。每个rank是一组内存颗粒。一个channel上的所有内存颗粒共享数据总线，所以同一个channel上有多个rank并不会增加内存带宽，它的作用更多的是增加内存密度。
每个内存颗粒都有一个CS#，即片选引脚。内存控制器能过将这个引脚置高来选择这个rank。
所以对于DDR3来说，rank是一个外部概念，但它通过片选信号还支持这个外部概念。
### Addressing

![[assets/Pasted image 20240613120743.png]]
## DDR4 

[jedec spec](assets/JESD79-4.pdf)

### Addressing

![[assets/Pasted image 20240613114903.png]]

# LPDDR


#  GDDR

RTX 4090有24颗 GDDR6X芯片(16Gb density)，总容量48GB，


GDDR就是更高速的DDR,

| ==Generation==                                             | CK# freq      | data rate in ref CK# | ==单引脚速率==            | 单芯片（DRAM)数据位宽 | 单芯片支持通道数 | 单芯片最大带宽 | 显卡可集成几块芯片         |
| ---------------------------------------------------------- | ------------- | -------------------- | -------------------- | ------------- | -------- | ------- | ----------------- |
| [GDDR5](https://en.wikipedia.org/wiki/GDDR5_SDRAM)         | 1~1.5GHz      | Quad                 | 4~6Gbps              | 16/32         | N.A.     | 24GB/s  | GTX 780集成16颗      |
| [GDDR5X](https://en.wikipedia.org/wiki/GDDR5_SDRAM#GDDR5X) | 1.125~1.25GHz | Octal(PAM4)          | 10~14Gbps            | 16/32         | N.A.     | 56GB/s  |                   |
| GDDR6                                                      |               | Octal(PAM4)          | 14Gbps\16Gbps\20Gbps | 16/32         |          | 64GB/   | RTX 2060有12颗16位芯片 |
| GDDR7                                                      |               |                      | **32**Gbps\48GBps    | 64            | 4        | 192GB/s |                   |
|                                                            |               |                      |                      |               |          |         |                   |
|                                                            |               |                      |                      |               |          |         |                   |
|                                                            |               |                      |                      |               |          |         |                   |
|                                                            |               |                      |                      |               |          |         |                   |
### GDDR5


| key                    | typical value                     | value range |
| ---------------------- | --------------------------------- | ----------- |
| **CK#时钟信号**：           | 1.25GHz                           | 1~1.5       |
| **WCK#时钟信号**：<br>      | 2.5GHz (二倍CK#)                    | 2~3         |
| **data rate per pin**： | 5Gbps(WCK#的上升沿和下降沿各采样一次)          | 4~6         |
| **单芯片数据引脚个数**:         | 32                                | 16/32       |
| **单芯片带宽**:             | 5Gbps * 32 = 160Gbps = 20GB/s<br> |             |
|                        |                                   |             |
|                        |                                   |             |
### 典型显卡解读

![[assets/Pasted image 20240613160621.png]]
 ![[assets/Pasted image 20240613160328.png]]
单芯片32位宽，总384位宽，算出芯片个数为12颗。即图中上、左、右各4颗。
总带宽288.4GB/s，算出单颗288.4/12=24GB/s。
单针脚带宽6Gbps，算出每颗芯片引脚数24GB/s / 6Gbps = 16个。
单针脚带宽6Gbps，则CK#频率1.25GHz, WK#频率2.5GHz。
### GDDR6


GDDR6分GDDR6和GDDR6X。GDDR6X在GDDR6的基础上，每个时钟周期传输的不再是2bits数据，而是4bits数据。原因是它在一个时钟的上升沿和下降沿各采样一次数据，一次采样的电平不再是两个电平，而是4个电平，即一次采样采2bits数据，这种技术叫PAM4。还有一种PAM3技术，是一个时钟周期传输3bits数据。

| key               | typical value                     |     |
| ----------------- | --------------------------------- | --- |
| **CK#时钟信号**：      | 1GHz                              |     |
| **WCK#时钟信号**：<br> | 2GHz (二倍CK#)                      |     |
| **DATA信号**：       | 16Gbps(WCK#的上升沿和下降沿各采样一次)         |     |
| **单芯片数据引脚个数**:    | 32                                |     |
| **单芯片带宽**:        | 5Gbps * 32 = 160Gbps = 20GB/s<br> |     |
|                   |                                   |     |
|                   |                                   |     |
### 典型显卡解读

RTX 2060数据：（[来源](https://www.techpowerup.com/gpu-specs/geforce-rtx-2060.c3310)）
![[assets/Pasted image 20240613162001.png]]
CK# 1.75GHz
WK# 3.5GHz
data rate per pin:  3.5GHz * 2(dual data rate) * 2(PAM4) = 14Gbps
per chip. bandwidth(16 pins): 14Gbps * 16 / 8bits per byte = 28GB/s
number of chips: 192/16 = 12
total bandwidth: 28GB/s * 12 = 336GB/s

## GDDR7
[Jedec Spec](assets/JESD239.01.pdf)

与DDR不同，GDDR把channel做为芯片内部概念，一个芯片有4个channel，一个channel有32个数据引脚，即单个芯片有128位宽。最高频率是32Gps，且每个时钟周期传输3bit数据。所以单颗GDDR7芯片的带宽为192GB/s。

主要参数：
![[assets/Pasted image 20240613112549.png]]
### GDDR7的单芯片192GB/s计算方法

根据提供的搜索结果,192GB/s这个带宽数值是针对GDDR7显存标准中每个单个GDDR7内存芯片的理论峰值带宽。具体来说:

- GDDR7标准支持每个引脚32Gbps的数据速率,规范最高可达48Gbps。[](https://ieeexplore.ieee.org/document/9731621)
- GDDR7采用PAM3编码方式,每2个时钟周期可传输3个比特,而非传统的2个比特。这使得GDDR7的数据传输效率提高了50%。[](https://pcr.cloud-mercato.com/providers/digitalocean/flavors/s-32vcpu-192gb)[](https://ieeexplore.ieee.org/document/9731621)
- GDDR7相比GDDR6,独立通道数从2个增加到4个。[](https://pcr.cloud-mercato.com/providers/digitalocean/flavors/s-32vcpu-192gb)[](https://ieeexplore.ieee.org/document/9731621)

因此,如果一个GDDR7芯片有32个数据引脚,每个引脚32Gbps,4个独立通道,则该芯片的理论带宽为:  
32 (引脚数) x 32Gbps (每引脚数据率) x 4 (通道数) x 1.5 (PAM3编码效率) = 192GB/s所以192GB/s是基于GDDR7标准中规定的最大配置时,单个GDDR7芯片可实现的理论峰值带宽。[](https://pcr.cloud-mercato.com/providers/digitalocean/flavors/s-32vcpu-192gb)[](https://ieeexplore.ieee.org/document/9731621)这一带宽数值相比上一代GDDR6有了翻番的提升,主要得益于PAM3编码、更高的数据速率和增加的独立通道数等技术创新。
# HBM

# PCIe

# NVLink