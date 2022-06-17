---

title: dbus开发调试工具和方法

---

# 概述

dbus是linux下提供跨进程服务的一套机制，类以于Android下的binder。不同的地dbus一般是基于unix domain socket或是Shared Memory Transport, named pipe等。

无论dbus底下是啥，它的效率是不如binder的。

## dbus底层的transport是什么

[transport spec](https://dbus.freedesktop.org/doc/dbus-specification.html#transports)

###  Unix Domain Socket(UDS)

略

###  launchd 

主要用于Mac OS X， 略。

### systemd 

### TCP Sockets

 tcp:host=127.0.0.1 or tcp:host=localhost.


## DBus主要概念

### Object 

object通常代表一个具体的应有，即每个应用有一个object。为了全局唯一，通常写作org.gnome.Nautilus这样的。

### Path

应用提供服务的路径，如果是本应用自定义的一些服务，路径就可以比较随意的写，比如说`/MainWindow`, `/test`等。
如果是本应用实现了其他应用定义的一些服务器接口，通常写作别的应用的路径，如/org/gnome/Nautilus。

### Interface

代表一组方法。通常写作org.gnome.XXXProvider之类的。

### Method

某Interface下的一个方法，就是正常的函数。

### 编址

｜A...	is identified by a(n)...	which looks like...	and is chosen by...
Bus	address	unix:path=/var/run/dbus/system_bus_socket	system configuration
Connection	bus name	:34-907 (unique) or com.mycompany.TextEditor (well-known)	D-Bus (unique) or the owning program (well-known)
Object	path	/com/mycompany/TextFileManager	the owning program
Interface	interface name	org.freedesktop.Hal.Manager	the owning program
Member	member name	ListNames	the owning program

## dbus工具

qdbusviewer, gui工具，用于查看系统中的所有object, path, inteface, method

dbus-monitor, commandline工具，用于实时显示bus上的request/reply。



## kdbus是啥

