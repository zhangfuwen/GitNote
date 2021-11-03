# Linux的打印等级
Linux有八个打印等级，以KERN_EMERG等级最高，KERN_DEBUG最低。


```c
#define KERN_EMERG       	"<0>"
#define KERN_ALERT			"<1>"
#define KERN_CRIT			"<2>"
#define KERN_ERR			"<3>" 
#define KERN_WARNNING		"<4>"
#define KERN_NOTICE			"<5>"
#define KERN_INFO			"<6>"
#define KERN_DEBUG			"<7>"
```

Linux系经中有一个当前打印级别，通常为4。这时打印级别大于4的打印内容都会被丢掉。

可以通过以下命令查看当前的打印等级：

```bash
cat /proc/sys/kernel/printk
4   4   1   7
```
其中第一个数这为当前的打印等级，这个默认值是在sysctl.conf中写的，在系统启动时就把这个值写到/proc/sys/kernel/printk这个文件了。也可以使用下面的命令修改其值:

```bash
echo 0 > /proc/sys/kernel/printk
cat /proc/sys/kernel/printk
0   4   1   7
```





