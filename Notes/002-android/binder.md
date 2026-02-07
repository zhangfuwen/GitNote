# binder基础知识

参考：https://github.com/wbo4958/jianshu/tree/master/docs/android-binder

![](/assets/res/2021-11-04-18-08-13.png)

binder_proc 在一个进程open("/dev/binder")时创建，并存储到filp->private_data中。
binder_proc代码一个打开/dev/binder的进程。
binder_proc记录了很多信息，包括但不限于：

1. task_struct, 从current获得。
2. todo, 这是一个待处理事务列表。对于服务端进程来讲，它是请求列表。对于客户端来讲，它是响应列表。

![](/assets/res/2021-11-04-18-10-43.png)

![](/assets/res/2021-11-04-18-12-30.png)
