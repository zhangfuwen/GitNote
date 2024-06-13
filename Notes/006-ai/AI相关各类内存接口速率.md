# 汇总

# DDR

## DDR3

[jedec spec](assets/JESD79-3E.pdf)

### Addressing


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