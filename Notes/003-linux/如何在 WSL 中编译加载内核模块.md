## 编译 WSL 内核
我们可以从微软的开源仓库中拉取最新的内核源码，然后自己编译内核。

拉取源码：

git clone https://github.com/microsoft/WSL2-Linux-Kernel.git
配置内核：

这里直接使用微软的默认配置：

mv Microsoft/config-wsl ./.config
编译：

make -j8

## WSL 启动新编译内核

将新编译好的内核拷贝到任意位置：

cp arch/x86_64/boot/bzImage /mnt/c/Users/uklar/
在启动 WSL 的用户的根目录中创建 WSL 配置文件：

touch /mnt/c/Users/uklar/.wslconfig
同时将如下内容写入配置文件：

[wsl2]
kernel=C:\\Users\\uklar\\bzImage
这里的配置就是指定 WSL 启动使用的内核。

关闭 WSL 再启动，以生效更改：

wsl --shutdown

![](assets/Pasted%20image%2020250422180329.png)

## 测试

hello.c 程序：

```c

#include <linux/init.h>
#include <linux/module.h>
#include <linux/printk.h>

static int __init hello_init(void)
{
    pr_info("Hello, world\n");
    return 0;
}

static void __exit hello_exit(void)
{
    pr_info("Goodbye, world\n");
}

module_init(hello_init);
module_exit(hello_exit);

MODULE_LICENSE("GPL");

```

```

Makefile：

```makefile

obj-m += hello.o

PWD := $(CURDIR)

all:
        make -C /home/uklar/WSL2-Linux-Kernel M=$(PWD) modules

clean:
        make -C /home/uklar/WSL2-Linux-Kernel M=$(PWD) clean

```

注意：这里要使用自己下载的内核源码的路径。

编译：

make
此时我们会得到 hello.ko

使用 sudo insmod hello.ko 测试是否可以加载模块，在 dmesg 命令的输出中我们可以看到：

[11114.649475] Hello, world
说明此时内核模块已经成功加载了。