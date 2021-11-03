# 描述性宏定义

内核代码中有一个描述性质的宏定义，它们会有出现在modinfo命令的打印结果里。

    modinfo netconsole.ko
    filename:       netconsole.ko
    license:        GPL
    description:    Console driver for network interfaces
    author:         Maintainer: Matt Mackall <mpm@selenic.com>
    srcversion:     4771B517D075072C76D13FE
    depends:        configfs
    intree:         Y
    vermagic:       3.2.0-29-generic SMP mod_unload modversions
    parm:           netconsole:
    netconsole=[src-port]@[src-ip]/[dev],[tgt-port]@<tgt-ip>/[tgt-macaddr] (string)
    
    modinfo macvlan.ko
    filename:       macvlan.ko
    alias:          rtnl-link-macvlan
    description:    Driver for MAC address based VLANs
    author:         Patrick McHardy <kaber@trash.net>
    license:        GPL
    srcversion:     703C9153A206A2631BF64D8
    depends:
    intree:         Y
    vermagic:       3.2.0-29-generic SMP mod_unload modversions
    
    version:        3.5.24-k2-NAPI

上述显示的内容分别由以下宏定义产生：

author:

```c
MODULE_AUTHOR("xxxxx")
```
description:

```c
MODULE_DESCRIPTION("yyyyyyyyyyyy");
```
license:

```c
MODULE_LICENSE("GPL");
```
alias:

```c
MODULE_ALIAS("alternate_name");
```

version:

```c
MODULE_VERSION(version_string)
```

parm:

```c
static char *string_test = “this is a test”;
static num_test = 1000;
module_param (num_test,int,S_IRUGO);
module_param (string_test,charp,S_ITUGO);
```


```c
#if LINUX_VERSION_CODE>=KERNULVERSION(2.4.9)
```
