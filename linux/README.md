# linux物理内存

一、 查看物理内存空间占用情况

    cat /proc/iomem

二、查看内核对内存的分配情况

    cat /proc/meminfo

三、查看系统内存余量

    free -m
    free -g
    free -t

四、查看进程的内存使用情况

    top

五、查看程序的虚拟地址占用情况

    cat /proc/{pid}/maps
