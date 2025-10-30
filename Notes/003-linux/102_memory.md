
![Linux中的Memory Compaction [一]](https://picx.zhimg.com/v2-0866738ee070d04e77f6ade81b9040de_720w.jpg?source=172ae18b)

Linux使用的是虚拟地址，以提供进程地址空间的隔离。它还带来另一个好处，就是像vmalloc()这种分配，不用太在乎实际使用的物理内存的分布是否**连续**，因而也就弱化了物理内存才会面临的「内存碎片」问题。

但如果使用kmalloc()，且申请的是[high order](https://zhuanlan.zhihu.com/p/105589621/edit)的内存大小（比如order为3，对应8个pages），则要求物理内存必须是连续的。在系统中空闲内存的总量（比如空闲10个pages）大于申请的内存大小，但没有连续的high order的物理内存时，我们可以通过**migrate**（迁移/移动）空闲的page frame，来**聚合**形成满足需求的连续的物理内存。

**【实现原理】**

来看下具体是如何操作的。假设现在有如下图所示的一段内存，其中有8个已使用的movable pages，8个空闲的free pages。

![](https://pic1.zhimg.com/v2-7a97afd6b66a9670de9547ea28b5e15e_1440w.jpg)

现在我们把左侧的4个movable pages和右侧的4个free pages交换位置，那么就会形成8个free pages全在左侧，8个movable pages全在右侧的情形。这时如果要分配order为3的内存就不是什么问题了。

![](https://pica.zhimg.com/v2-e8348daa3beac34e2887506430820dd0_1440w.jpg)

这样的一套机制在Linux中称为**memory compaction**， "compaction"中文直译过来的意思是「压实」，这里free pages就像是气泡一样，migration的过程相当于把这些气泡挤压了出来，把真正有内容的pages压实了。同时，这也可以看做是一个归纳整理的过程，因此"memory compaction"也被译做「内存规整」。

在此过程中，需要两个**page** **scanners**，一个是"migration scanner"，用于从source端寻找可移动的页面，然后将这些页面从它们所在的LRU链表中isolate出来；另一个是"free scanner"，用于从target端寻找空闲的页面，然后将这些页面从buddy分配器中isolate出来。

![](https://pic3.zhimg.com/v2-5eb6d5c618d73567f776882bfe7a6e68_1440w.jpg)

那一个page frame被migrate之后，使用这段内存的应用程序还怎么访问这个page的内容呢？这就又要归功于Linux中可以将**物理实现**与**逻辑抽象**相分离的虚拟内存机制了，程序访问的始终是虚拟地址，哪怕物理地址在migration的过程中被更改，只要修改下进程页表中对应的[PTE项](https://zhuanlan.zhihu.com/p/67053210)就可以了。虚拟地址保存不变，程序就“感觉”不到。

**【使用限制】**

由于要保持虚拟地址不变，像[kernel space](https://zhuanlan.zhihu.com/p/68501351)中线性映射的这部分物理内存就不能被migrate，因为线性映射中虚拟地址和物理地址之间是固定的偏移，migration导致的物理地址更改势必也造成虚拟地址的变化。也就是说，并非所有的page都是“可移动”的，这也是为什么[buddy系统](https://zhuanlan.zhihu.com/p/105589621)中划分了"migrate type"。

![](https://pic3.zhimg.com/v2-e815da27d12ce47f51786da53b2f57d4_1440w.jpg)

**【何时启动】**

在无法满足high order的内存分配时，可进行类似于"[direct reclaim](https://zhuanlan.zhihu.com/p/72998605)"的**direct compaction**，但它们都面临同样的问题：内存分配的过程被阻塞，增加分配等待的时间。

除了满足**当前的**high order的内存需求，内存的compaction机制还肩负着在buddy系统的基础上进一步减少内存碎片，为**将来的**high order的内存需求做好准备。那是不是应该稍微积极一点，像kswapd那样，在一定条件下被激活。

一种设想是干脆把内存compaction工作直接交给kswapd来完成，反正目的都是为了腾出更多可用的内存，在reclaim的时候把compact也做了。可是kswapd要做的事已经很多，这样会进一步增加kswapd的复杂性。

那就单独创建一个和kswapd类似的内核线程，专门负责"**background compaction**"，这就是per-node的**kcompated**线程的雏形。

虽然compaction操作可以减少内存碎片，但其过程涉及到内存的拷贝和页表的重新建立，具有相当的开销，因而应该尽力避免，所以同kswapd一样，也是需要时才激活。kswapd是根据[zone watermark](https://zhuanlan.zhihu.com/p/73539328)的值被唤醒的，那kcompactd呢？可以像[内存的writeback](https://zhuanlan.zhihu.com/p/71217136)机制一样，每隔一段时间唤醒，具体的讨论请参考[这个patch](https://link.zhihu.com/?target=https%3A//lore.kernel.org/patchwork/patch/575291/)。

此外，提前准备好的high order内存，因而没有马上投入使用，可能在接下来的内存分配中又被**split**掉了，那之前的内存compaction岂不是白做了，所以引入background compaction后，是否需要对这些high order内存进行保护也是需要考虑的问题。

[Linux中的Memory Compaction [二] - CMA](https://zhuanlan.zhihu.com/p/105745299)

  

**参考：**

- [LWN - Page migration](https://link.zhihu.com/?target=https%3A//lwn.net/Articles/157066/)
- [LWN - Memory compaction](https://link.zhihu.com/?target=https%3A//lwn.net/Articles/368869/)
- [LWN - Proactive compaction for the kernel](https://link.zhihu.com/?target=https%3A//lwn.net/Articles/817905/)

  

_原创文章，转载请注明出处。_