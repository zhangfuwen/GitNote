# 内核模块与驱动程序
我们通常以内核模块的方式开发驱动程序，但驱动程序并不一定非要是一个内核模块，内核模块也不一定非要驱动一个硬件。这两者是各自独立的概念。

我的同事就曾写过一个内核模块，将这个模块插入到内核中时可以实现文件操作的统计之类的功能。而我有时候也会把一些跟设备交互的程序代码直接编译进内核里。

使用内核模块的方式开发驱动程序的好处在于你可以不必编程整个内核，也不必频繁的重启机器。我们可以像安装运行普通程序一样，把一段代码放进内核空间里去运行。当然，写得不好的内核模块仍然可能导致系统崩溃或挂死，因而可能的话，我一般都先在应用层去验证一下某些设备寄存器的操作是否会产生预期的结果。

应用层驱动程序及如何在应用层对一些功能进行验证我会单列一个章节。这里先介绍内核模块，以便如果有初学者学习这本书的话可以先找到一些成就感（其实就我而言，我认为应用层的驱动开发和调试技能属于高级技能，初学者还是先学学普通的驱动开发方式好了，呵呵）。
## 内核模块
一个内核完整的内核模块(.ko文件)至少需要两个函数：入口函数（俗称init函数）和出口函数（俗称exit函数）。当用户通过insmod xxx.ko的方式插入一个内核模块到内核空间时，init函数会被调用，init函数是内核模块运行的开始点。当用户通过rmmod xxx的方式移除一个内核模块时，exit函数会被调用，exit函数是内核模块的退出点。

init和exit函数是对一个内核模块的入口函数和出口函数的一般称呼，而不是一个规定好了的,必须使用的函数名。它们的函数名可以由开发者自己定义，但传入参数和返回值的格式却是规定好了的。开发者还必须通过module_init和module_exit两个宏告之编译器init和exit函数的名字。

下面是一个完整可编译的内核模块的代码，它是我从网上随便找到的，但归跟结底的来源是LDD3.
    #include <linux/init.h>
    #include <linux/module.h>
    MODULE_LICENSE("Dual BSD/GPL");

    static int hello_init(void)
    {
        printk(KERN_ALERT "Hello, world\n");
        return 0;
    }

    static void hello_exit(void)
    {
        printk(KERN_ALERT "Goodbye, cruel world\n");
    }

    module_init(hello_init);
    module_exit(hello_exit);

与上面讲的一样，这段代码有两个基本函数，hello_init和hello_exit，然后用module_init和module_exit宏来声明哪个是入口函数，哪个是出口函数。hello_init和hello_exit两个函数名中的init和exit字样完全没有用，如果你自己不会记混的话，完全可以将init函数取名为hello_exit，如果那样的话，你的代码就是这样的：

    #include <linux/init.h>
    #include <linux/module.h>
    MODULE_LICENSE("Dual BSD/GPL");

    static int hello_exit(void)
    {
        printk(KERN_ALERT "Hello, world\n");
        return 0;
    }

    static void hello_init(void)
    {
        printk(KERN_ALERT "Goodbye, cruel world\n");
    }

    module_init(hello_exit);
    module_exit(hello_init);
两个函数的返回值一个是int，一个是void。init函数return 0表示成功插入模块。有些模块在init函数中做对系统环境或硬件函境做一个检测，如果发现系统环境或硬件函数不允许自己正确运行的话，它就返回一个负值，从而阻止自己运行下去。

函数声明然的static关键字不是必须的，static关键字声明的函数不允许在该函数外面调用，如果你想象一个c文件是C++程序中的一个类的话，以static方式声明的c函数相当于C++中的private函数成员。你经常可以看到内核模块声明为static。

printk函数是内核提供的api，它的使用方式与C语言中的printf完全一样(我遇到过的只有一点不一样，即%f不工作)。

KERN_ALERT是一个宏定义，它基本等价于"<1>"。因而printf(KERN_ALERT "hello world!")，等价于printk("<1>hello world")。有些人不知道printf("xxxx" "yyyy")这样的写法，这样的写法比较蛋疼，但却是合法的，相当于printf("xxxxyyyy"),没试过的可以试试。

Linux内核本身在正常情况下不应该打印东西的，它只做为一个后台服务员默默奉献。但有些时候它也需要记录一事情，比如说某个程序运行出错了，某个设备不正常了之类，方便系统管理员修复或开发人员调试程序。

Linux内核将自己要记录的事情分了八个等级。以一个数字标识，越小的越重要。如果你以printk("<7>hello")这样的方式打印，打印内容很可能被直接丢掉，因为这样的打印一点也不重要。

    MODULE_LICENSE("Dual BSD/GPL");
MODULE_LICENSE这个宏声明这个内核代码发行时遵循的协议。通常对于中国的开发者来说这句没什么实际意义，照抄就好，不写问题也不大，插入时内核会抱怨一句kernel tainted而已。

至于那两个include包含的头文件，这里面的两个是必须，任何内核模块都必须包含。
## Makefile
要想编译这个内核模块你需要有你现在正在运行的Linux系统的内核源代码，以及一个Makefile文件。
Makefile可以如下编写：

    obj-m := hello.o
    KERNELDIR := /usr/src/linux-headers-3.0.11-generic
    PWD := $(shell pwd)

    modules:
        $(MAKE) -C $(KERNELDIR) M=$(PWD) modules
    clean:
        rm -rf *.o *~ core .depend .*.cmd *.ko *.mod.c

上面的Makefile文件有一个地方需要修改才能在你的机器上编译，KERNELIDR := 后面的路径要替换成你的系统中的内核源代码所在的路径。

## 获得内核源代码
要获得你的OS的内核原代码，ubuntu系统下可以通过以下命令获取：

    sudo apt-get install linux-source

如果你使用的是其他Linux发行版，你可以百度一下：［你的发行版] 安装内核源码，或者安装ubuntu，后者是最好的办法。

## 编译与测试
以上准备就绪后通过以下命令编译和插入模块来查看结果：

    make  #这时你应该能在当前目录下看到hello.ko文件
    insmod hello.ko #这时你应该能看到init函数中的打印消息
    lsmod | grep hello #lsmod命令列出当前系统中的所有内核模块
    rmmod hello.ko  #这时你应该能看到exit函数中的打印消息

