---

title: netstat
tags: ['netstat', 'network']

---

# 安装

# 使用

```bash
netstat -apn
```

p代表显示进程的名称和pid
a代表以数字形式展示，因为有时dns查询比较耗时。

a表示所有socket类型，其他可选值：

     <Socket>={-t|--tcp} {-u|--udp} {-U|--udplite} {-S|--sctp} {-w|--raw}
           {-x|--unix} --ax25 --ipx --netrom

简单来说还可以用：t,u,x

# 显示监听列表

```bash
netstat -l

# 或

netstat --listening
```

# 显示路由表

```bash
netstat -r

# 或

netstat --route

```

# 显示网络接口

```bash
netstat -i
netstat --interfaces
```

显示效果类似于：

    Iface      MTU    RX-OK RX-ERR RX-DRP RX-OVR    TX-OK TX-ERR TX-DRP TX-OVR Flg
    lo       65536  2606621      0      0 0       2606621      0      0      0 LRU
    lxcbr0    1500        0      0      0 0             0      0      0      0 BMU
    wlp0s20f  1500  9469656      0      0 0       4284188      0      0      0 BMRU

可能没啥用。


