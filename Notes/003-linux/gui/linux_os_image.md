---

title: linux系统镜像
tags: ['image', 'bootimage']

---

# 从网络下载镜像的同时制作启动优盘

正常人制在启动盘的方法是先下载，再dd到优盘。但这样比较慢。我们可以找两个步骤串起来，可以节省大量时间。

## 从ssh服务器dd

```bash
ssh ssh_host_name "dd if=path_to_your_image | gzip -1 - " | gunzip -1 - | sudo dd iflag=fullblock of=/dev/sdd oflag=direct,sync status=progress bs=1M

```

这个脚本使用了压缩，可以节省网络带宽。

## 从http服务器下载并dd

```bash
curl 'http://address.for/your.iso' | dd conv=noerror,sync ifag=fullblock oflag=direct,sync bs=1M of=/dev/sdX
```

如果服务器需要用户名密码则写为：

```bash
curl -u username:passwd 'http://address.for/your.iso' | dd conv=noerror,sync ifag=fullblock oflag=direct,sync bs=1M of=/dev/sdX
```
