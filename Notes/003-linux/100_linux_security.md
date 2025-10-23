## [Linux 提权原理及十种提权姿势详解](https://zhuanlan.zhihu.com/p/304572787)

作者：廖林 Ivens  
链接：https://zhuanlan.zhihu.com/p/304572787  
来源：知乎  
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。  
  

这篇文章摘录至[《渗透测试方法论之 Linux 提权》](https://link.zhihu.com/?target=https%3A//www.lanqiao.cn/courses/2650)，目前该课程已上线蓝桥教育平台，其中梳理了 Linux 提权的十种方法，并且每个提权场景都会搭配相应的**在线实战实验**，能帮助你更好的理解提权的原理，掌握提权的方法和技巧。

课程制作非常不易，共花费了我十多个周末来修修改改，课程价格实惠就几十块钱，如果对你有帮助的话，就请支持我一下吧，真心的感谢！

> _课程上线首周有八折优惠，加上我的专属优惠码（7MaDpzNc）还能折上折哦！_

[![](https://pic1.zhimg.com/v2-32470dbd9f5fc74fa4a7b71a196fa9f2_120x160.jpg)渗透测试方法论之 Linux 提权实战www.lanqiao.cn/courses/2650](https://link.zhihu.com/?target=https%3A//www.lanqiao.cn/courses/2650)

![](https://pic3.zhimg.com/v2-0b68d3f93cc8dbd9afbbe899b85bbb1a_1440w.jpg)

在开始学习 Linux 提权之前，我们需要先理解什么是提权？为什么要进行提权？

## 0x1 什么是提权？

一次完整的渗透测试流程，应该包含以下 8 个步骤：

![](https://picx.zhimg.com/v2-1c11312a874244bb8ce20e2cc6ec2525_1440w.jpg)

在渗透测试前期，我们会想尽办法通过「某个漏洞攻击」获取到目标主机系统上的一个 shell 用于执行系统命令，从而达到控制目标主机的目的，也就是图中的第4步。

当我们获取到的初始 shell 为低权限用户时，需要想办法提升权限为高权限用户，这个过程就叫做系统权限提升，简称「提权」 （对应图中第5步）。

## 0x2 为什么 shell 会有高低权限的区别？

有些同学会有一个疑惑：为什么同样是通过漏洞攻击获取到的 shell ，有时候是低权限，有时候又是高权限呢？

下面以一个简单的例子来说明：

![](https://pic3.zhimg.com/v2-209d5efd3f2834d30c8d3c9075fa5db2_1440w.jpg)

通常情况下 Web 服务架构如上图所示：**后端 linux 服务器通过 Tomcat 中间件提供 Web 服务**。

如今的计算机操作系统都是基于多用户设计的，通常一台计算机上同时存在着多个不同权限的系统用户，而在 Linux 服务器上，每个服务都需要先赋予某个用户权限才能运行。

> 用户权限定义:「用户权限」是用来控制某个系统用户被允许做哪些操作和不允许做哪些操作。

假设 tomcat 服务以普通用户 `web-user`的权限运行并提供 Web 服务，此时如果黑客通过 Web 漏洞获取到 shell，此 shell 就是一个 `web-user`用户权限的 shell——此时就需要提权。

反之，若 tomcat 是以 root 用户权限运行，那黑客获取到的 shell 就是 root 权限。

## 0x3 为什么要提权？

在渗透测试中，高权限 shell 能带来以下好处：

- 高权限能对更多的文件进行「增删改查」操作，便于进一步收集主机系统中的敏感信息
- Linux 系统的 root 权限可获取 shadow 文件中的密码 Hash，若内网环境中存在「账户/密码复用」的情况，可用于横向扩展
- Windows 系统中的 system 权限可提取内存中的密码 Hash，可进一步用于域渗透

简而言之，获取更高权限的 shell，能为渗透测试后期的工作带来便利。

### 0x4 Linux 提权之暴力破解提权

> 摘录至蓝桥教育平台[《渗透测试方法论之 Linux 提权》](https://link.zhihu.com/?target=https%3A//www.lanqiao.cn/courses/2650)第十二章，内容会有所删减。

在前面的实验中，我们已经为大家介绍了多种 Linux 提权方法，这些方法已经覆盖到了实战中的绝大部分场景，但如果你遇到前面所有的方法都无法成功提权时，别忘了还有一种最原始但有用的方法可以尝试——**暴力破解**。

通常来说，可以通过两种途径暴力破解 root 用户的密码：

1. 通过 su 命令爆破 root 密码
2. 通过 SSH 协议爆破 root 密码

下面我依次为大家介绍这两种方法。

## 4.1 通过 su 命令暴破 root 密码

Linux su 命令用于切换为其他使用者身份，除 root 外，其他用户使用时需要输入将要切换的「目标用户」的密码。

su 命令暴力破解使用的工具是 `sucrack` 。

### sucrack 介绍和安装

sucrack 是一个多线程的`Linux`工具，用于通过 su 爆破本地用户密码。

因为 su 命令需要从 TTY shell 获取用户输入，因此不能只用一个简单的 shell 脚本来完成爆破，sucrack 是采用 c 语言编写的工具，并且支持多线程，爆破效率非常高。

sucrakck 官方地址如下：

```text
https://leidecker.info/projects/sucrack.shtml
```

![](https://pic1.zhimg.com/v2-f5b73aa98c29dd9a0f52376f90c341e8_1440w.jpg)

在实战环境中，靶机可能是无法连接外网的，因此我们可以使用如下两种方法安装 sucrack:

1. 下载`sucrack`源码并上传到靶机上，再编译运行。
2. 下载`sucrack`源码，在本地编译好之后再上传到靶机运行。

本课程旨在为大家演示 `sucrack` 的用法，因此为了方便我们直接使用 apt 安装 sucrack ：

```text
 sudo apt install sucrack
```

![](https://pic1.zhimg.com/v2-a2ff40991577151ff4eacc7c036ce22c_1440w.jpg)

如上图所示，安装成功。

### sucrack 使用方法

sucrack 的用法非常简单，最基础的命令如下：

```text
sucrack -w 20 wordlists.txt
```

参数解释：

- `-w` 指定线程数
- `wordlists.txt` 爆破使用的字典

sucrack 默认爆破 root 用户，你也可以使用 `-u` 指定要爆破的用户:

```text
sucrack -u myuser -w 20 wordlists.txt
```

我预先准备了一份密码字典用于演示，存放在 `/tmp/common-wordlists.txt`，同学们也可以使用自己的字典。

![](https://pic3.zhimg.com/v2-e5bbcef12de402fac7af27b4b62a7cd8_1440w.jpg)

使用如下命令尝试爆破 root 密码，线程数保守一点设定为 20：

```text
sucrack -w 20 /tmp/common-wordlists.txt
```

开始爆破之后，按键盘任意键刷新显示进度:

![](https://pic4.zhimg.com/v2-ac5d7285b04e49d2885a217c74146a79_1440w.jpg)

等待一会，成功破解出 root 用户的密码为 ****ly:

![](https://pic1.zhimg.com/v2-6a5dc318239e07d8ae4c85078a606c36_1440w.jpg)

然后使用 `su root` 切换为 root 用户：

![](https://picx.zhimg.com/v2-f8aad1254cef301440de4d0e40ff90a7_1440w.png)

sucrack 功能是非常简单强大的，你学会了吗？

## 4.2 通过 SSH 爆破 root 密码

### SSH 服务介绍

SSH 服务的配置文件有为两个，分别是：

- /etc/ssh/ssh_config : 客户端的配置文件
- /etc/ssh/sshd_config : 服务端的配置文件

仅当 `/etc/ssh/sshd_config` 中 `PermitRootLogin` 设置为 yes，root 用户才能登录 ssh：

![](https://pic2.zhimg.com/v2-40956e43115fd8e92054ae968615d0fd_1440w.jpg)

因此，在通过 ssh 爆破 root 用户密码之前，我们需要先使用如下命令来查询靶机是否允许 root 用户通过 ssh 登录：

> 注意，此时我们已经获取到目标靶机的初始 shell，因此可以直接查看靶机上的 sshd 配置文件，后续的 ssh 爆破也是在本地进行的。

```text
cat /etc/ssh/sshd_config |grep -i permitrootlogin
```

![](https://pic2.zhimg.com/v2-83bd736b8dcab0729888c6bbd1ddd647_1440w.png)

当 PermitRootLogin 被设置为 `yes` 时，才用尝试 SSH 暴力破解，否则就没必要在尝试了。

### hydra 介绍和使用方法

SSH 协议爆破使用到的工具是 hydra，hydra 是一款非常著名的爆破工具，除了 SSH 协议以外，hydra 还支持众多其他协议例如 RDP、SMB、HTTP、MySQL 等，由于篇幅有限，具体可以参考下面这篇文章：

```text
https://github.com/Jewel591/OSCP-Pentest-Methodologies/blob/master/Password%20Attacks/README.md
```

hydra 在 Kali Linux 上默认安装，实验主机的 ubuntu 环境也已经提前安装好，爆破 SSH 协议的语法如下：

```text
hydra -l root -P passwords.txt -t 4 -V <IP> ssh
```

参数解释：

- `-l` : 指定用户名
- `-P` : 指定爆破字典
- `-t` : 指定爆破线程
- `-V` : 查看破解详情、进度

下面我们尝试爆破 root 用户的密码，字典仍然使用 `/tmp/common-wordlists.txt` ：

```text
hydra -l root -P /tmp/common-wordlists.txt -t 64 -V 127.0.0.1 ssh
```

![](https://pic1.zhimg.com/v2-ff520c9d36f1dd845822b285f9c93480_1440w.jpg)

如上图所示，成功爆破出 root 用户的密码。

注意：如果你使用的 hydra 是 v9.0 之前的版本，在爆破 ssh 协议时很可能会出现误报，请先升级后再使用（实验环境是 v9.2）。

## 4.3 本章总结

在本节实验中，我们学习了通过两种途径爆破 SSH 服务的方法—— `su` 和 `SSH`，并且分别介绍了对应的两种破解工具的使用。

更多提权技巧详解，请见[《渗透测试方法论之 Linux 提权》](https://link.zhihu.com/?target=https%3A//www.lanqiao.cn/courses/2650)。