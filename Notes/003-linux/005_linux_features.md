
## [Linux 内核中有哪些比较牛逼的设计?](https://www.zhihu.com/question/332710035/answer/1854780284)

作者：水灵木子珈  
链接：https://www.zhihu.com/question/332710035/answer/1854780284  
来源：知乎  
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。  
  

cgroup + [namespace](https://zhida.zhihu.com/search?content_id=367313283&content_type=Answer&match_order=1&q=namespace&zhida_source=entity).

基于这两个技术，有了[LXC](https://zhida.zhihu.com/search?content_id=367313283&content_type=Answer&match_order=1&q=LXC&zhida_source=entity)也有了[Docker](https://zhida.zhihu.com/search?content_id=367313283&content_type=Answer&match_order=1&q=Docker&zhida_source=entity)。

简单的来说，两个技术的作用就是：

- **cgroup = limits how much you can use;**
- **namespaces = limits what you can see (and therefore use**

## **cgroup:**

cgroups 全称control group，用来限定一个进程的资源使用，由Linux 内核支持，可以限制和隔离Linux进程组 (process groups) 所使用的物理资源 ，比如cpu，内存，磁盘和网络IO，是Linux container技术的物理基础。

### cgroups 层级结构（Hierarchy）

内核使用 cgroup 结构体来表示一个 control group 对某一个或者某几个 cgroups 子系统的资源限制。cgroup 结构体可以组织成一颗树的形式，每一棵cgroup 结构体组成的树称之为一个 cgroups 层级结构。cgroups层级结构可以接手一个或者几个 cgroups 子系统，当前层级结构可以对其 接手的cgroups 子系统进行资源的限制，而每一个 cgroups 子系统只能被一个 cpu 层级结构接手。

![](https://picx.zhimg.com/50/v2-58d55062621c682d9eb9571632aeb056_720w.jpg?source=2c26e567)

上图表示两个cgroups层级结构，每一个层级结构中是一颗树形结构，树的每一个节点是一个 cgroup 结构体（比如cpu_cgrp, memory_cgrp)。第一个 cgroups 层级结构接手了 cpu 子系统和 cpuacct 子系统( _cpuacct_子系统 （CPU accounting）会自动生成报告来显示 cgroup中 任务所使用的 CPU 资源，其中包括子群组任务)， 当前 cgroups 层级结构中的 cgroup 结构体就可以对 cpu 的资源进行限制，并且对进程的 cpu 使用情况进行统计。 第二个 cgroups 层级结构接手了 memory 子系统，当前 cgroups 层级结构中的 cgroup 结构体就可以对 memory 的资源进行限制。

在**每一个 cgroups 层级结构中，每一个节点（cgroup 结构体）可以设置对资源不同的限制权重**。比如上图中 cgrp1 组中的进程可以使用60%的 cpu 时间片，而 cgrp2 组中的进程可以使用20%的 cpu 时间片。

那么，**内核又是如何将cgroups同进程绑定的呢**？

下面这个图从整体结构上描述了进程与 cgroups 之间的关系。最下面的`P`代表一个进程。每一个进程的描述符中有一个指针指向了一个辅助数据结构`css_set`（cgroups subsystem set）。图中的”M×N Linkage”说明的是`css_set`通过辅助数据结构可以与 cgroups 节点进行多对多的关联。

![](https://pica.zhimg.com/50/v2-932d616c3750f2a60083f872eee73d27_720w.jpg?source=2c26e567)

  

一个进程只能隶属于一个`css_set`，一个`css_set`可以包含多个进程，隶属于同一`css_set`的进程受到同一个`css_set`所关联的资源限制。cgroups 的实现不允许`css_set`同时关联同一个cgroups层级结构下多个节点。 这是因为 cgroups 对同一种资源不允许有多个限制配置。

## Namespace:

namespace另一个维度的资源隔离技术，大家可以把这个概念和我们熟悉的C++和Java里的namespace相对照。如果CGroup设计出来的目的是为了隔离上面描述的物理资源，那么namespace则用来隔离PID(进程ID),IPC,Network等系统资源。我们现在可以将它们分配给特定的Namespace，每个Namespace里面的资源对其他Namespace都是透明的。不同container内的进程属于不同的Namespace，彼此透明，互不干扰。

![](https://pica.zhimg.com/50/v2-4231d9ffeff898bad704926ab3fe8dfe_720w.jpg?source=2c26e567)

  

  

推荐资料：

[cgroup源码分析1—— css_set和cgroup的关系](https://link.zhihu.com/?target=http%3A//linux.laoqinren.net/kernel/cgroup-source-css_set-and-cgroup/)