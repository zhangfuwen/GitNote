# x86的启动过程

# 1.1 CPU的第一条指令
借用网友的一张图片，上面讲述了x86系统启动到Linux内核的流程。从图中可以看到，CPU上电后从CS:IP=FFFF:0000这地址处获取第一条指令并执行。

![X86架构下的从开机到Start_kernel启动的总体过程](/assets/20140807152559130.jpeg)

这条指令即是BIOS的指令，但它并不是BIOS的第一条指令，反而是最后一条指令。这个指令通常是一个跳转指令，跳转到BIOS中合适的位置去执行。

需要知道这时CPU处于实模式（实模式、保持模式、系统管理模式），实模式下，CPU采用段式管理内存，CPU访存的物理地址由段选择寄存器CS和指令指针寄存器IP决定。两个寄存器都是16位寄存器，一个16位的地址只能访问2的16次方即64KB字节的内存范围，但是在8086中，地址总线已经扩展到20位，即它最多可以访问1MB的内存。为了组成20位的地址，CPU将CS这个16位寄存器左移4位，与IP寄存器相加做为地址，相加得到的数是一个20位的数。在CS的值的不变的情况下，CPU只能访问CS到CS+64KB之间的地址，如果想访问超过这个范围的地址，就需要修改CS的值。这种访存模式就好人一个人牵着一个小狗，人不动的情况下，小狗运行的距离超不过那跟绳子。
在我们现在讨论的环境中，地址CS：0xFFFF，IP：0x即物理地址0xFFFF0。

# 1.2 现代计算机中的CPU第一条指令

在现代CPU中，BIOS存储在一块SPI接口的flash中。从网上找了一张简单的架构图。这个图比较过时了，然而仍然可以用来解释这个问题。

在现在计算机中，北桥基本已经不存在了，或者说这个北桥的功能已经被合并到CPU Die上面去了。所以大家可以看到的只是南桥。从图中可以看出，北桥的功能主要是提供PCIe接口用来接入独立显卡，或是在北桥内部集成一个显卡，北桥的功能非常单一，单独搞成一个芯片不划算，又给画电路板带来不必要的麻烦，所以直接把这个芯片合并在CPU Die上是非常合理的。

去掉北桥之后，CPU通过DMI总线与南桥通信。这里要大致介绍一下Intel CPU中的总线系统。Intel CPU的系统总线称为QPI，它由20个lane组成，每个lane是一对差分传输线。Intel的外部总线是PCIe总线，PCIe可以说是QPI的 一个子集，它与QPI采用类似的上传协议，在物理层支持4,8,16,32等多种个数的lane。DMI总线几乎只用在CPU和南桥之间，它由4个lane组成，它只实现了总线的底层协议，对上层协议来讲，它是透明的。如果在CPU和南桥之间通过PCIe连接，那么南桥上的设备就挂这个PCIe总线的下面，但通过DMI总线连接时，南桥上的设备也是挂在系统了CPU_BUS_NO0下面。南桥上有LPC总线控制器和SPI总线控制器，SPI总线上接的就是存储BIOS的flash。在另一些系统中，SPI总线控制器也会挂在LPC总线上。

所有这些总线上传输的都是CPU发出来的load/store访存指令（对于x86来讲，类似于一个mov汇编指令）或in/out 访问IO指令。各级总线的控制器可以理解为一个桥，它接收来自上级总线的访存请求，转换成下级总线（如spi)的对应一个或多个transaction，得到数据后再返回给上级总线。

这里先不谈IO指令，只谈访存，在不同的总线上，总线地址代表的含义是不一样的。比如说通常使用的8MB SPI flash。在SPI总线上，它的地址范围是0~8MB。对CPU来讲，这个8MB的数据被映射为到4GB地址的末尾8MB。0xFFFF0这个物理地址上的数据，在spi flash上可能存储的位置是实际是0xFFFFFFF0(4GB-16Bytes)，对这个地址的数据读请求路由到SPI总线时，SPI控制器会做一个转换，转换为0x7FFFF0地址（8MB-16Bytes)，发给spi flash，取回数据，再给CPU回复一个memory read response。由于QPI是异步总线，所以SPI总线慢它可以等。

![](/assets/cd306921220602561.png)

说CPU取的第一条指令地址是0xFFFF0，这是针对8086而言，对于80386之后的CPU，这个地址就变成了0xFFFFFFF0。

这里可能需要细讲一个CPU的段寄存器，CPU的段寄存器有几个，CS, DS, ES, FS，即代码段寄存器、数据段寄存器等。CPU执行的第一条指令的地址是由CS寄存器和IP寄存器共同指定的。在CPU上电后，CPU内部一些硬件逻辑做执行一次自检，检查自己的状态是否正常，如果正常的话，将eax寄存器写为0.不正常则会写入错误编码。
然后CPU清空Cache, TLB等缓存类硬件，将CS的初始值设置为0xF000，将IP设置为0xFFF0,所得到的物理地址值即为0xF000 << 4 | 0xFFF0 = 0xFFFF0。

看起来已经很明白了，然而还没完。
现代CPU中为了加速指令地址的计算，为每个段寄存器增加了两个寄存器：Base和Limit。Base存放基址，Limit存放最大偏移值。Base和Limit寄存器不能通过指令直接读写，他们的值是在写段寄存器时由CPU自动设置的。通常Base等于段寄存器左移四位，如果CS的值为0xF000，CS的Base寄存器则为0xF0000，但CPU初始化时例外。CS的值为0xF000, 但其Base为0xFFFF0000，EIP为0xFFF0,此时对应的指令地址为0xFFFF0000+0xFFF0 = 0xFFFFFFF0。0xFFFFFFF0就是CPU将要执行的第一条指令。这造成这样一个有趣的事实，16位程序眼中的指令地址空间0x0000~0xFFFF（大小为64K）被CPU翻译到物理地址空间(0xFFFF0000~0xFFFFFFFF)。也就是说，从CPU初始化，到段寄存器被重写（通过跨段跳转指令）前，指令空间0x0000~0xFFFF通过段寄存器被映射到物理地址空间0xFFFF0000~0xFFFFFFFF。 

参考资料：
http://www.cppblog.com/djxzh/archive/2015/07/12/uefi_resetvector.html