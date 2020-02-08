# 字符设备驱动
字符设备驱动是最简单的一种设备驱动程序，它与用户之间通过简单文件操作接口来交互数据，驱动开发者**直接处理**来自应用层的read/write/ioctl等请求，而不是经过内核提供的一些**高层抽象**。如果你实现的是一个块设备的话，你处理的请求并不是直接来自于用户的，而是经过内核的bio模块处理（分解或合并）过的。如果你实现的是一个网卡设备，你处理的数据并不是直接来自于用户的tcp或udp接口，而是经过tcp/ip协议栈处理过的socket buffer。

下面的代码并不是一段可用的代码，只是用来描述一个字符设备驱动的基本结构。一个字符设备驱动需要三样东西，(1) 设备号，(2) cdev结构体，(3) file_operations结构体。


```c
#include <linux/module.h>
#include <linux/types.h>
#include <linux/fs.h>
#include <linux/errno.h>
#include <linux/mm.h>
#include <linux/sched.h>
#include <linux/init.h>
#include <linux/cdev.h>
#include <asm/io.h>
#include <asm/system.h>
#include <asm/uaccess.h>
```


```c
#define MEMDEV_MAJOR 251   /*预设的mem的主设备号*/
#define MEMDEV_NUM   2     /*设备数*/

struct mem_dev
{
     unsigned   int size;
     char   *data;
     struct semaphore sem;
};

static int mem_major = MEMDEV_MAJOR;
struct cdev mem_cdev;
struct mem_dev *mem_devp;

static const struct file_operations mem_fops = {
     .owner= THIS_MODULE,
     .open= mem_open,
     .write= mem_write,
     .read= mem_read,
     .release= mem_release,
     .llseek= mem_llseek,
};

static int __init memdev_init(void)
{
    int result;
    int err;
    dev_t devno = MKDEV(mem_major, 0);
    if(mem_major) {
        result= register_chrdev_region(devno, MEMDEV_NUM, "memdev");
    }else{
        result= alloc_chrdev_region(&devno, 0, MEMDEV_NUM, "memdev");
        mem_major= MAJOR(devno);
    }
    if(result< 0) {
        printk("can'tget major devno:%d\n", mem_major);
        return result;
    }

    cdev_init(&mem_cdev,&mem_fops);
    mem_cdev.owner= THIS_MODULE;

    err= cdev_add(&mem_cdev, MKDEV(mem_major, 0), MEMDEV_NUM);
    if(err) {
        printk("add cdev faild,err is %d\n", err);
    }

    return result;

fail_malloc:
    unregister_chrdev_region(MKDEV(mem_major,0), MEMDEV_NUM);
    returnresult;
}

static void memdev_exit(void)
{
    cdev_del(&mem_cdev);
    unregister_chrdev_region(MKDEV(mem_major,0), MEMDEV_NUM);
    printk("memdev_exit\n");
}

module_init(memdev_init);
module_exit(memdev_exit);

MODULE_LICENSE("GPL");
```

### 设备号
设备号用来标识一个设备，它分为两部分：主设备号和次设备号。主设备号是标识一个设备所属的分类，次设备号在这个分类中标识一个唯一的设备。可以通过MKDEV宏定义将主设备号和次设备号组装成一个完整的dev_t类型的设备号。

```c
dev_t my_dev_no = MKDEV(主设备号，次设备号);
```
或通过MAJOR和MINOR宏定义从一个dev_t类型的设备号中提取数主设备号和次设备号。

```c
unsigned int major = MAJOR(my_dev_no);
unsigned int minor = MINOR(my_dev_no);
```

以上三个宏定义定义于linuc/kdev.h中。

内核中一些常用类别的设备是有特定的主设备号的，但字符设备没有。驱动开发者可以自己找一个没有被使用的主设备号，但在一个系统中这个主设备号没有被使用不代表在另一个系统中它也没有被使用。所以靠谱的方式是设内核在运行时自动给你分配一个主设备号。但这样又存在一个问题是你不知道内核给你分配的是什么号。两者都有缺点。

开发者通过register_chrdev_region来注册一个自己设定的设备号，如果该函数返回负值则表明这个预设的设备号已被占用。
开发者通过alloc_chrdev_region来请求内核为自己自动分配一个设备号，如果返回值为负值则表明自动分配失败。

```c
result= register_chrdev_region(devno, MEMDEV_NUM, "memdev");
result= alloc_chrdev_region(&devno, 0, MEMDEV_NUM, "memdev");
```
两个函数的都可以一次分配主设备号相同，次设备号连续的多个设备号。参数MEMDEV_NUM为你一次要分配的设备号的个数。入参的最后一个参数可以传入开发者定义的任意字符串，这个字符串可以在procfs或sysfs中显示给用户。

对于register_chrdev_region，开发者需要传入自己预设的设备号devno。对于alloc_chrdev_region，开发者传入一个dev_t结构体的指针，内核会将自动分配得到的设备号放入这个指针指定的位置。alloc_chrdev_region的第二个参数是开发者指定的从设备号起始值。假设开发者传入的是0, 并且MEMDEV_NUM的值为3, 则开发者会得到三个主设备号相同，从设备号依次为0,1,2的设备号。

无论是register_chrdev_region还是alloc_chrdev_region得到的设备号，它们都通过unregister_chrdev_region来释放。

```c
unregister_chrdev_region(MKDEV(mem_major,0), MEMDEV_NUM);
```

设备号唯一标识一个设备，用户可以利用主次设备号，通过mknod命令创建一个**设备文件**。设备文件是/dev/目录或其子目录下的一个文件，对这个文件的读写等操作最终会转化为对设备驱动程序的read/write函数的调用。

## file_operations结构体
设备驱动向用户提供了一个设备文件，对设备文件的读写操作对应着设备驱动程序的read/write函数调用。设备驱动程序通过file_operations结构体向内核提供这些函数。

```c
static const struct file_operations mem_fops = {
    .owner= THIS_MODULE,
    .open= mem_open,
    .write= mem_write,
    .read= mem_read,
    .release= mem_release,
    .llseek= mem_llseek,
};
```
上述代码中的mem_open, mem_write, mem_read, mem_release, mem_llseek是开发者提供的函数。向**设备文件**写入数据时，mem_write就会被调用。.owner = THIS_MODULE是固定用法，不必特殊关注。

以.open = mem_open这样的形式初始化一个结构体是gcc支持特殊用法，c99以后的C标准也支持这种写法。这种写法允许程序员只填充结构体的特定成员，没有提及到的成员变量会被编译器自动填充为0。

file_operations结构体中的open, write, read 等函数的传入参数及返回值格式在2.2节中详细介绍。

## cdev结构体
cdev结构体是对**字符设备**的抽象。通过cdev_add将一个cdev结构体填加到内核中，即代表将一个字符设备加到系统中。

```c
err= cdev_add(&mem_cdev, MKDEV(mem_major, 0), MEMDEV_NUM);
if(err) {
    printk("add cdev faild,err is %d\n", err);
}
```
从代码中可以看到，填加cdev时，还要同时提供设备号和设备个数。假设MEMDEV_NUM值为3, 则上述代码相当于向系统填加了三个字符设备，它们的主设备号是mem_major，次设备号依次为0,1,2。

向内核填加cdev结构之前需要先初始化它的成员变量。cdev结构体有很多成员变量，其中很多可以填充为固定的非0值。在这种情况下，这个成员变量不能不理会，每次都一个一个去填充又太麻烦，因而内核提供了一个cdev_init函数，这个函数将cdev结构体的成员变量填充为固定的值，然后开发者主需要填充自己关注的就可以了。

```c
cdev_init(&mem_cdev,&mem_fops);
mem_cdev.owner= THIS_MODULE;
```
从代码上可以看出, file_operations结构体要做为传入参数引入。

在删除模块时，需要通过cdev_del来从内核中删除这个结构体。

```c
cdev_del(&mem_cdev);
```
