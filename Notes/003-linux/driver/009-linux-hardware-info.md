# 查看Linux硬件系统

## GUI工具


### hardinfo

```
sudo apt install hardinfo
```

![](assets/hardinfo.png)



## command line

### lspci

```bash
sudo apt install pciutils
```

```
➜  ~ lspci -tv
-[0000:00]-+-00.0  Intel Corporation 10th Gen Core Processor Host Bridge/DRAM Registers
           +-01.0-[01]--+-00.0  NVIDIA Corporation TU117M [GeForce GTX 1650 Ti Mobile]
           |            \-00.1  NVIDIA Corporation Device 10fa
           +-02.0  Intel Corporation CometLake-H GT2 [UHD Graphics]
           +-04.0  Intel Corporation Xeon E3-1200 v5/E3-1500 v5/6th Gen Core Processor Thermal Subsystem
           +-08.0  Intel Corporation Xeon E3-1200 v5/v6 / E3-1500 v5 / 6th/7th/8th Gen Core Processor Gaussian Mixture Model
           +-12.0  Intel Corporation Comet Lake PCH Thermal Controller
           +-14.0  Intel Corporation Comet Lake USB 3.1 xHCI Host Controller
           +-14.2  Intel Corporation Comet Lake PCH Shared SRAM
           +-14.3  Intel Corporation Comet Lake PCH CNVi WiFi
           +-16.0  Intel Corporation Comet Lake HECI Controller
           +-1c.0-[04-51]----00.0-[05-51]--+-00.0-[06]----00.0  Intel Corporation JHL7540 Thunderbolt 3 NHI [Titan Ridge 4C 2018]
           |                               +-01.0-[07-2b]--
           |                               +-02.0-[2c]----00.0  Intel Corporation JHL7540 Thunderbolt 3 USB Controller [Titan Ridge 4C 2018]
           |                               \-04.0-[2d-51]--
           +-1c.7-[54]----00.0  Realtek Semiconductor Co., Ltd. RTS525A PCI Express Card Reader
           +-1d.0-[55]----00.0  Toshiba Corporation XG6 NVMe SSD Controller
           +-1f.0  Intel Corporation Device 068e
           +-1f.3  Intel Corporation Comet Lake PCH cAVS
           +-1f.4  Intel Corporation Comet Lake PCH SMBus Controller
           \-1f.5  Intel Corporation Comet Lake PCH SPI Controller

```

### lscpu

```bash
sudo apt install util-linux
```

```
➜  ~ lscpu
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Byte Order:                      Little Endian
Address sizes:                   39 bits physical, 48 bits virtual
CPU(s):                          12
On-line CPU(s) list:             0-11
Thread(s) per core:              2
Core(s) per socket:              6
Socket(s):                       1
NUMA node(s):                    1
Vendor ID:                       GenuineIntel
CPU family:                      6
Model:                           165
Model name:                      Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz
Stepping:                        2
CPU MHz:                         800.075
CPU max MHz:                     5000.0000
CPU min MHz:                     800.0000
BogoMIPS:                        5199.98
Virtualization:                  VT-x
L1d cache:                       192 KiB
L1i cache:                       192 KiB
L2 cache:                        1.5 MiB
L3 cache:                        12 MiB
NUMA node0 CPU(s):               0-11
Vulnerability Itlb multihit:     KVM: Mitigation: VMX disabled
Vulnerability L1tf:              Not affected
Vulnerability Mds:               Not affected
Vulnerability Meltdown:          Not affected
Vulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:        Mitigation; Enhanced IBRS, IBPB conditional, RSB filling
Vulnerability Srbds:             Not affected
Vulnerability Tsx async abort:   Not affected
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clfl
                                 ush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm con
                                 stant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpui
                                 d aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma 
                                 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes
                                  xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_sin
                                 gle ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid
                                  ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx 
                                 smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln 
                                 pts hwp hwp_notify hwp_act_window hwp_epp pku ospke md_clear flush_l1d arch_
                                 capabilities

```


### lsusb

```
sudo apt install usbutils
```

```
➜  ~ lsusb -t    
/:  Bus 04.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/2p, 10000M
    |__ Port 2: Dev 3, If 0, Class=Hub, Driver=hub/2p, 5000M
/:  Bus 03.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/2p, 480M
/:  Bus 02.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/10p, 10000M
/:  Bus 01.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/16p, 480M
    |__ Port 1: Dev 2, If 1, Class=Human Interface Device, Driver=usbhid, 12M
    |__ Port 1: Dev 2, If 0, Class=Human Interface Device, Driver=usbhid, 12M
    |__ Port 5: Dev 14, If 0, Class=Hub, Driver=hub/6p, 480M
        |__ Port 1: Dev 15, If 0, Class=Human Interface Device, Driver=usbhid, 12M
        |__ Port 1: Dev 15, If 1, Class=Human Interface Device, Driver=usbhid, 12M
        |__ Port 5: Dev 16, If 0, Class=Human Interface Device, Driver=usbhid, 480M
    |__ Port 6: Dev 18, If 2, Class=Vendor Specific Class, Driver=, 480M
    |__ Port 6: Dev 18, If 0, Class=Wireless, Driver=rndis_host, 480M
    |__ Port 6: Dev 18, If 1, Class=CDC Data, Driver=rndis_host, 480M
    |__ Port 8: Dev 6, If 3, Class=Video, Driver=uvcvideo, 480M
    |__ Port 8: Dev 6, If 1, Class=Video, Driver=uvcvideo, 480M
    |__ Port 8: Dev 6, If 4, Class=Application Specific Interface, Driver=, 480M
    |__ Port 8: Dev 6, If 2, Class=Video, Driver=uvcvideo, 480M
    |__ Port 8: Dev 6, If 0, Class=Video, Driver=uvcvideo, 480M
    |__ Port 9: Dev 8, If 0, Class=Vendor Specific Class, Driver=, 12M
    |__ Port 14: Dev 9, If 0, Class=Wireless, Driver=btusb, 12M
    |__ Port 14: Dev 9, If 1, Class=Wireless, Driver=btusb, 12M

```

### lshw

```bash
sudo apt install lshw lshw-gtk
```
[lshw.log](assets/lshw.log)


## sysstat

system performance tools for Linux
The sysstat package contains the following system performance tools:

 - sar: collects and reports system activity information;
 - iostat: reports CPU utilization and disk I/O statistics;
 - tapestat: reports statistics for tapes connected to the system;
 - mpstat: reports global and per-processor statistics;
 - pidstat: reports statistics for Linux tasks (processes);
 - sadf: displays data collected by sar in various formats;
 - cifsiostat: reports I/O statistics for CIFS filesystems.


 ```bash
 sudo apt install sysstat
 ```