---

title: url scheme handler

---


# 什么是url scheme handler?

在操作系统中，不同的应用之间可以相互拉起，但又不知道自己拉起的应用到底是谁。

比如说 你想拉起邮件客户端，但又不确定系统里安装了哪个邮件客户端，比如说你可能有thunderbird, 也可能是outlook。如何能无差别地拉起一个邮件客户端？

你可以这样:

```bash
xdg-open mailto:zhangfuwen@foxmail.com
```

mailto:zhangfuwen@foxmail.com就是一个url scheme。

系统里有一个映射关系，这个映射关系可能通过任何方式维护，在不同的操作系统可能不一样。或者同一个操作系统中也存在不同的机制。

# Linux

## 通过desktop file声明

```desktop
[Desktop Entry]
Version=1.0
Type=Application
Name=dnfurl
Exec=dnfurl %U
Terminal=false
NoDisplay=true
MimeType=x-scheme-handler/dnf
```

https://blog.kagesenshi.org/2021/02/custom-url-scheme-handler.html

显示每次查一下所有的desktop文件是不太可能的，其实OS会维护一个cache，可以通过命令更新：

```bash
cd /usr/share/applications/
update-desktop-database
```

## 直接修改cache文件

所谓的cache文件可能就是下面的几个文件：

```
/usr/share/applications/defaults.list
~/.local/share/applications/mimeapps.list
~/.config/mimeapps.list
```

我们修改文件可以直接达到效果：

```bash

sudo su

echo "x-scheme-handler/ddg=org.gnome.Totem.desktop" >> /usr/share/applications/defaults.list

xdg-open ddg:asdfadfl
```

以上代码可以打开totem，当然了，totem并不认识ddg：asdfadf1是什么，所以它会报错。

## 应用如何响应呢？

如果我们自己写一个应用，建立了映射关系，我们如何收到这个请求呢？

其实它就是做为命令行参数传给我们的应用的。

我们看看totem的desktop file:

```
➜  Code head /usr/share/applications/org.gnome.Totem.desktop
[Desktop Entry]
Name=Videos
Comment=Play movies
Exec=totem %U
Icon=org.gnome.Totem
DBusActivatable=true
````

可以看到Exec字段跟的是`totem %U`，这里的'%U'就是url list的意思，这个可以参考：https://specifications.freedesktop.org/desktop-entry-spec/desktop-entry-spec-latest.html#exec-variables

