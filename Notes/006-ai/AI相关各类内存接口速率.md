# 汇总

# DDR


| Generation | 时钟频率              | MT/s(clock rate的二倍) | burst chop | burst length | per chip data width |
| ---------- | ----------------- | ------------------- | ---------- | ------------ | ------------------- |
| DDR3       | 800MHz ~  1600MHz | 1600MT/s ~ 3200MT/s | 4          | 8            | x4, x8, x16         |
| DDR4       | 1066MHz ~ 2666MHz | 2133MT/s ~ 5333MT/s | 4          | 8            | x4, x8, x16         |
| DDR5       | 2000MHz ~ 4000MHz | 4GT/s ~ 8GT/s       | 8          | 16           | x32+x8 ECC          |


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


GDDR就是更高速的DDR,

| ==Generation== | ==单引脚速率==            | 单芯片（DRAM)数据位宽 | 单芯片支持通道数 | 单芯片最大带宽 | 显卡可集成几块芯片 |
| -------------- | -------------------- | ------------- | -------- | ------- | --------- |
| GDDR6          | 14Gbps\16Gbps\20Gbps | 32            |          |         |           |
| GDDR7          | **32**Gbps\48GBps    | 64            | 4        | 192GB/s |           |
|                |                      |               |          |         |           |
|                |                      |               |          |         |           |
## GDDR7
[Jedec Spec](assets/JESD239.01.pdf)

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