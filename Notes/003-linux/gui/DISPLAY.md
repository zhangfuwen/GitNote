---

title: Linux下的DISPLAY环境变量以及wayland equivalent

---

# X下的display

[an old book](assets/graphics_admin_guide.pdf)

DISPLAY环境变量的格式分三部分：

`xxx:display[.screen]`

格式为：[图片参考链接](https://books.google.com.hk/books?redir_esc=y&hl=zh-CN&id=FNBQAAAAYAAJ&dq=shmlink&focus=searchwithinvolume&q=shmlink)

[display name format](assets/display_name_format.png)



如localhost:0.0, 192.168.1.1:0.0, unix:0.0, shmlink:0.0, 

1. xxx

xxx不是`unix`, `shmlink`, 'local`, 或省略的情况下，认为是主机名或ip地址，走的是tcpip协议，端口为6000+display。

xxx是unix的情况下，用的是unix domain socket(UDS), 就是`/tmp/.X11-unix/X0`或`@/tmp/.X11-unix/X0`，其中0是`display`部分指定的。

xxx是shmlink的情况下，走的是Shared Memory Transport(SMT)。

xxx是local或省略的情况，用库自动选择当前机器上最高效的协议，如果有SMT就走SMT，如果没有使能SMT就走UDS，实在不行走TCP。

2. display

一个机器可能有多个display，这个是逻辑上的概念，通常是一个物理的，多个虚拟的。

一个display包含多个screen。如果display是物理的，那么screen就代表你机器上的真实显示器。

3. screen如上，可省略。


# wayland






