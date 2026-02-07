# file_operations结构体

file_operations结构体是字符设备驱动的核心。对设备文件任何文件式的操作都会转换为对file_operations结构体中的成员函数的调用。
C语言结构体没有成员函数的概念，这里所谓的成员函数实际上是一些类型为函数指针的成员变量。
file_operations结构体定义于linux/fs.h中，其主要成员如下：

```c
struct file_operations {
    struct module *owner;
    loff_t  (*llseek) (struct file * filp, loff_t pos, int);
    ssize_t  (*read) (struct file *filep, char __user * buf, size_t len, loff_t * pos);
    ssize_t  (*aio_read) (struct kiocb *, char __user *, size_t, loff_t);
    ssize_t  (*write) (struct file *filp, const char __user *buf, size_t len, loff_t * pos);
    ssize_t  (*aio_write) (struct kiocb *, const char __user *, size_t, loff_t);
    int (*readdir) (struct file *, void *, filldir_t);
    unsigned int (*poll) (struct file *, struct poll_table_struct *);
    int (*ioctl) (struct inode *, struct file *, unsigned int cmd, unsigned long data);
    int (*mmap) (struct file *, struct vm_area_struct *);
    int (*open) (struct inode *, struct file *);
    int (*flush) (struct file *);
    int (*release) (struct inode *, struct file *);
    int (*fsync) (struct file *, struct dentry *, int datasync);
    int (*aio_fsync) (struct kiocb *, int datasync);
    int (*fasync) (int, struct file *, int);
    int (*lock) (struct file *, int, struct file_lock *);
    ssize_t  (*readv) (struct file *, const struct iovec *, unsigned long, loff_t *);
    ssize_t  (*writev) (struct file *, const struct iovec *, unsigned long, loff_t *);
    ssize_t  (*sendfile) (struct file *, loff_t *, size_t, read_actor_t, void __user *);
    ssize_t  (*sendpage) (struct file *, struct page *, int, size_t, loff_t *, int);
    unsigned long (*get_unmapped_area) (struct file *, unsigned long,
                                        unsigned long, unsigned long,
                                        unsigned long);
};
```

## struct file和 struct inode
在介绍file_operations之前先介绍两个结构体，struct file 和 struct inode, 只列出可能用到的成员。

```c
struct file {
    const struct file_operations    *f_op; //这个就是你传进去的file_operations结构体
    void            *private_data;         //该成员是系统调用时保存状态信息非常有用的资源。

    unsigned int         f_flags;          //表示文件打开时采用的标志:
                                           //O_RDWR,O_RDONLY,O_SYNC,O_NBLOCK

    mode_t            f_mode;              //文件权限，可读，可写，可执行
                                           //chmod +x file时会改变该位

    loff_t            f_pos;               //文件指针，指示当文件被读或写到什么位置了，
                                           //除了llseek函数外，其他函数不应该改变这个字段。

    struct dentry *f_dentry;               //文件路径，通常驱动开发用不到它，
                                           //唯一用到它的时候就是通过它来找到inode结构体
                                           //filp->f_dentry->d_inode
}

struct inode {
    dev_t            i_rdev;                //设备号，对于注册多个设备的驱动程序可以根
                                            //子设备号来知道要访问的是哪个设备
                                            //这个字段在不同的版本间可能会有变化，可以通过
                                            //imajor和iminor两个函数来得到主次设备号，这样
                                            //的代码是可移植的。

    struct cdev *i_cdev;                    //cdev结构体提针，有些时候我们把这个cdev结构体放
                                            //在另一个更大的mydev结构中，这时可以通过
                                            //container_of宏来得到mydev结构
}

unsigned int iminor(struct inode *inode);
unsigned int imajor(struct inode *inode);
```


下面按常用程度的顺序依次介绍。

## open

open函数有些时候并不是必须的，特别是你只有一个同类设备的时候。

```c
int (*open) (struct inode *, struct file *);
```
用户打开一个设备文件时调用open，入参struct file和struct inode分别描述打开的**设备文件**的文件描述符(file descritpor, fd)和这个**设备文件**的文件信息。

## read

```c
ssize_t (*read) (struct file * filp, char _ _user * buf, size_t len, loff_t * pos);
```
这个函数要求驱动程序从pos偏移处读到len个字节到buf中。pos偏移具体代表什么意义由驱动开发者定义，它也可以没有意义，这时可以忽略这个参数。

## write

```c
ssize_t  (*write) (struct file *filp, const char __user *buf, size_t len, loff_t * pos);
```
这个函数要求驱动程序将buf中的len字节数据写入到设备的pos偏移处。pos偏移具体代表什么意义由驱动开发者定义，它也可以没有意义，这时可以忽略这个参数。

## ioctl

```c
int (*ioctl) (struct inode *, struct file *, unsigned int cmd, unsigned long data);
```

这个接口用来发送一些read/write不太容易做到的设备相关的命令。它的后两个参数cmd和data的意义完全由驱动开发者去定义，通常cmd会传递命令类型，data会传递命令所有附带的数据。data可以是一个long型的数据，也可以传递一个用户空间的缓冲区指针。如果传入的是用户空间的缓冲区指针，则需要驱动代码中用copy_from_user和copy_to_user来对它进行操作。

## llseek

```c
loff_t (*llseek) (struct file *, loff_t offset, int whence);
```
该函数移动文件读写指针到指定的位置，offset是移动的长度，whence指定是相对位移还是决对位移。

whence可选值是SEEK_SET（绝对位移，移到距文件开头offset处), SEEK_CURR（相对位移，移动距当前位置offset处）， SEEK_END（相对位移，移到距文件结尾-offset处, offset应为负值）。

llseek的一个实现示例如下：

```c
loff_t scull_llseek(struct file *filp, loff_t off, int whence)
{
    struct scull_dev *dev = filp->private_data;
    loff_t newpos;

    switch(whence)
    {
    case 0: /* SEEK_SET */
            newpos = off;
            break;

    case 1: /* SEEK_CUR */
            newpos = filp->f_pos + off;
            break;

    case 2: /* SEEK_END */
            newpos = dev->size + off;
            break;

    default: /* can't happen */
            return -EINVAL;
    }
    if (newpos < 0)
            return -EINVAL;
    filp->f_pos = newpos;
    return newpos;
}
```

