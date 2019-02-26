# 设备号
设备号结构dev_t及其相关的宏定义都在linux/kdev.h中。
下面这段话出自linux内核linux/kdev.h中，是关于主次设备号操作的一些宏

    #ifndef _LINUX_KDEV_T_H
    #define _LINUX_KDEV_T_H
    #ifdef __KERNEL__
    #define MINORBITS 20
    #define MINORMASK ((1U << MINORBITS) - 1)

    #define MAJOR(dev) ((unsigned int) ((dev) >> MINORBITS))
    #define MINOR(dev) ((unsigned int) ((dev) & MINORMASK))
    #define MKDEV(ma,mi) (((ma) << MINORBITS) | (mi))

从上述代码中可以看出，dev_t本质是一个unsigned int。它的低20位存放次设备号，高位存放主设备号。MKDEV(ma,mi) 就是先将主设备号左移20位，然后与次设备号相加得到设备号。

MAJOR(dev)得到的是dev的高12位，MINOR(dev)得到的是dev的低20位。


