# 中断系统
由于Linux最早是面向x86开发的，其中断系统依然保留着一些x86架构特有的称谓，比如说**irq**。在早期的x86系统中，中断是由中断控制器8259来上报的，8259有8个中断源引脚，两个8259级联（通常的用法）共有15个中断源。这15个中断源也意味着当时的x86系统只有15个外部中断向量，它们依次被称为irq0~irq15。CPU本身会产生一些**异常**，比如说除数为0异常等，这些异常又占用了一些中断向量，通常是0〜31。因而真正的irq（8259)所能使用的中断向量是32~46。从x86架构的原有含义来讲,irq号并不是**中断向量**号，irq号标识的是中断控制器上的中断源。irq号与中断向量号之间存在线性偏移关系。但Linux将irq号等同于中断向量号, Linux代码里讲的irq已经不是x86架构下的irq的概念了。

##irq_desc
Linux用一个irq_desc结构体来描述一个中断向量的相关信息。利用irq_desc[]数组来描述所有的中断向量。在多核或多CPU系统中，每个逻辑CPU都可以各自独立地处理中断向量，因而系统中可以会存在多个irq_desc[]，即irq_desc[]是一个percpu的数组。

    struct irq_desc irq_desc[NR_IRQS] __cacheline_aligned_in_smp = {
        [0 ... NR_IRQS-1] = {
            .handle_irq = handle_bad_irq,
            .depth      = 1,
            .lock       = __RAW_SPIN_LOCK_UNLOCKED(irq_desc->lock),
        }
    };

irq_desc描述的信息包括这个中断向量的**通用处理函数**，中断控制器，中断处理函数列表等。
一个中断被解发后，会运行一个通用中断处理函数handle_irq，但系统中可能用多个硬件会触发同一个中断向量，即这个irq可能是共享的(IRQF_SHARED)，这时系统需要这个通用处理函数遍历中断处理函数列表*action，运上面的每个中断处理函数。*actiong列表上的元素就是驱动开发者调用request_irq时挂上去的。*action本身是一个单向链表结构体，由next指针指向下一个操作，因此action实际上是一个操作链，可以用于共享IRQ线的情况。
![http://rock3.info/blog/2013/11/17/irq_desc%E6%95%B0%E7%BB%84%E7%9A%84%E5%88%9D%E5%A7%8B%E5%8C%96%E8%BF%87%E7%A8%8B/](http://rock3.info/wp-content/uploads/2013/11/irq_desc.jpg)



