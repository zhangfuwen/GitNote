# 中断系统

由于Linux最早是面向x86开发的，其中断系统依然保留着一些x86架构特有的称谓，比如说**irq**。在早期的x86系统中，中断是由中断控制器8259来上报的，8259有8个中断源引脚，两个8259级联（通常的用法）共有15个中断源。这15个中断源也意味着当时的x86系统只有15个外部中断向量，它们依次被称为irq0~irq15。CPU本身会产生一些**异常**，比如说除数为0异常等，这些异常又占用了一些中断向量，通常是0〜31。因而真正的irq（8259)所能使用的中断向量是32~46。从x86架构的原有含义来讲,irq号并不是**中断向量**号，irq号标识的是中断控制器上的中断源。irq号与中断向量号之间存在线性偏移关系。但Linux将irq号等同于中断向量号, Linux代码里讲的irq已经不是x86架构下的irq的概念了。

##irq_desc
Linux用一个irq_desc结构体来描述一个中断向量的相关信息。利用irq_desc[]数组来描述所有的中断向量。在多核或多CPU系统中，每个逻辑CPU都可以各自独立地处理中断向量，因而系统中可以会存在多个irq_desc[]，即irq_desc[]是一个percpu的数组。
```c
    struct irq_desc irq_desc[NR_IRQS] __cacheline_aligned_in_smp = {
        [0 ... NR_IRQS-1] = {
            .handle_irq = handle_bad_irq,
            .depth      = 1,
            .lock       = __RAW_SPIN_LOCK_UNLOCKED(irq_desc->lock),
        }
    };
```
irq_desc描述的信息包括这个中断向量的**通用处理函数**，中断控制器，中断处理函数列表等。
一个中断被解发后，会运行一个通用中断处理函数handle_irq，但系统中可能用多个硬件会触发同一个中断向量，即这个irq可能是共享的(IRQF_SHARED)，这时系统需要这个通用处理函数遍历中断处理函数列表*action，运上面的每个中断处理函数。*actiong列表上的元素就是驱动开发者调用request_irq时挂上去的。*action本身是一个单向链表结构体，由next指针指向下一个操作，因此action实际上是一个操作链，可以用于共享IRQ线的情况。
![http://rock3.info/blog/2013/11/17/irq_desc%E6%95%B0%E7%BB%84%E7%9A%84%E5%88%9D%E5%A7%8B%E5%8C%96%E8%BF%87%E7%A8%8B/](http://rock3.info/wp-content/uploads/2013/11/irq_desc.jpg)

# 使用workqueue
中断处理函数运行于临界段，不能运行一个可能出现调度的代码，比如说不能使用锁。
workqueue,中文称其为工作队列，是一个用于创建内核线程的接口，通过它工作队列（workqueue）是另外一种将工作 推后执行的形式。工作队列可以把工作推后，交由一个内核线程去执行，也就是说，这个下半部分可以在进程上下文中执行。 这样，通过工作队列执行的代码能占尽进程上下文的所有优势。最重要的就是工作队列允许被重新调度甚至是睡眠。
那么，什么情况下使用工作队列，什么情况下使用tasklet。如果推后执行的任务需要睡眠，那么就选择工作队列。如果推后执行的任务不需要睡眠，那么就选择tasklet。另外，如果需要用一个可以重新调度的实体来执行你的下半部处理，也应该使用工作队列。它是唯一能在进程上下文运行的下半部实现的机制，也只有它才可以睡眠。这意味着在需要获得大量的内存时、在需要获取信号量时，在需要执行阻塞式的I/O操作时，它都会非常有用。如果不需要用一个内核线程来推后执行工作，那么就考虑使用tasklet。
## 2.6.20版本以前的用法
```c
    // prepare
    struct workqueue_struct * myworkqueue = create_workqueue("myworkqueue");

    //finish
    flush_workqueue(myworkqueue);
    destroy_workqueue(myworkqueue);

    //put something to work
    void my_func(void *data)
    {
        char *name = (char *)data;
        printk(KERN_INFO “Hello world, my name is %s!\n”, name);
    }
    struct work_struct my_work;
    INIT_WORK(&my_work, my_func, “Jack”);
    queue_work(my_workqueue, &my_work);
```
以下代码用到的函数原型为：



```c
    struct workqueue_struct *create_workqueue(const char *name)
    int queue_work(struct workqueue_struct *wq, struct work_struct *work)
    int queue_delayed_work(struct workqueue_struct *wq, struct work_struct *work, unsigned long delay)
    void flush_workqueue(struct workqueue_struct *wq)
    void destroy_workqueue(struct workqueue_struct *wq)
```



##2.6.20之后的用法
2.6.20之后的版本没有办法再通过work_struct的data字段传递函数参数，而需要将work_struct嵌入到自己定义的数组结构（这个数组结构里有你想要传递的参数）里，然后通过container_of宏来得到自定义的数组结构。



```c
    typedef void (*work_func_t)(struct work_struct *work);
    struct work_struct {
        atomic_long_t data;
        struct list_head entry;
        work_func_t func;
    };
```


使用方法：



```c
    //defination
    struct my_struct {
        int data_to_pass;
        struct work_struct thework;
    }
    void my_func(struct work_struct *work)
    {
        struct my_struct *mydata = container_of(work, struct my_struct, thework);
        printk(KERN_INFO “Hello world, my data is %d!\n”, mydata->data_to_pass);
    }

    // prepare
    struct workqueue_struct * myworkqueue = create_workqueue("myworkqueue");

    //finish
    flush_workqueue(myworkqueue);
    destroy_workqueue(myworkqueue);

    //put something to work
    struct my_struct my_data;
    my_data.data_to_pass = 5;
    INIT_WORK(&(my_data.my_work), my_func);
    queue_work(my_workqueue, &my_data.my_work);
```



**长按识别二维码或手机扫描二维码
打赏我1.5元**
![](/assets/mm_facetoface_collect_qrcode_1486597617608.png)

