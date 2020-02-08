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


