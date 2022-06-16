---
title: dbus开发调试工具和方法
---
# 概述

dbus是linux下提供跨进程服务的一套机制，类以于Android下的binder。不同的地dbus一般是基于unix domain socket或是Shared Memory Transport, named pipe等。

无论dbus底下是啥，它的效率是不如binder的。

## dbus底层的transport是什么

[transport spec](https://dbus.freedesktop.org/doc/dbus-specification.html#transports)

### Unix Domain Socket(UDS)

略

### launchd

主要用于Mac OS X， 略。

### systemd

### TCP Sockets

tcp:host=127.0.0.1 or tcp:host=localhost.

## DBus主要概念

### Service

Service通常代表一个具体的应有，即每个应用有一个Service为了全局唯一，通常写作org.gnome.Nautilus这样的。

### Object/Path

应用提供服务的路径，如果是本应用自定义的一些服务，路径就可以比较随意的写，比如说`/MainWindow`, `/test`等。
如果是本应用实现了其他应用定义的一些服务器接口，通常写作别的应用的路径，如/org/gnome/Nautilus。

### Interface

代表一组方法。通常写作org.gnome.XXXProvider之类的。

标准接口(Standard Interfaces)

每个dbus object都要提供的接口，用于dbus manager获取关于这个object的信息。

#### org.freedesktop.DBus.Peer 获取object是否存活、uuid

```
org.freedesktop.DBus.Peer.Ping ()
org.freedesktop.DBus.Peer.GetMachineId (out STRING machine_uuid)
```


#### org.freedesktop.DBus.Introspectable 

获取object提供的其他interface、method信息

```javascript
 org.freedesktop.DBus.Introspectable.Introspect (out STRING xml_data)
```

#### org.freedesktop.DBus.Properties

```javascript
org.freedesktop.DBus.Properties.Get (in STRING interface_name,
                                               in STRING property_name,
                                               out VARIANT value);
org.freedesktop.DBus.Properties.Set (in STRING interface_name,
                                   in STRING property_name,
                                   in VARIANT value);
org.freedesktop.DBus.Properties.GetAll (in STRING interface_name,
                                      out ARRAY of DICT_ENTRY<STRING,VARIANT> props);
org.freedesktop.DBus.Properties.PropertiesChanged (STRING interface_name,
                                ARRAY of DICT_ENTRY<STRING,VARIANT> changed_properties,
                                ARRAY<STRING> invalidated_properties);
```

#### org.freedesktop.DBus.ObjectManager

子对象管理

```javascript
org.freedesktop.DBus.ObjectManager.GetManagedObjects (out ARRAY of DICT_ENTRY<OBJPATH,ARRAY of DICT_ENTRY<STRING,ARRAY of DICT_ENTRY<STRING,VARIANT>>> objpath_interfaces_and_properties);
org.freedesktop.DBus.ObjectManager.InterfacesAdded (OBJPATH object_path, ARRAY of DICT_ENTRY<STRING,ARRAY of DICT_ENTRY<STRING,VARIANT>> interfaces_and_properties);
org.freedesktop.DBus.ObjectManager.InterfacesRemoved (OBJPATH object_path, ARRAY<STRING> interfaces);
```



### Method

某Interface下的一个方法，就是正常的函数。

### 编址

| A...	      | is identified by a(n)...	 | which looks like...                           | 	and is chosen by...                                           |
|------------|---------------------------|-----------------------------------------------|----------------------------------------------------------------|
| Bus        | 	address                  | 	unix:path=/var/run/dbus/system_bus_socket    | 	system configuration                                          |
| Connection | 	bus name                 | 	:34-907 (unique) or com.mycompany.TextEditor | (well-known)	D-Bus (unique) or the owning program (well-known) |
| Object     | 	path                     | 	/com/mycompany/TextFileManager               | 	the owning program                                            |
| Interface  | 	interface                | name                                          | 	org.freedesktop.Hal.Manager	the owning program                |
| Member     | 	member name              | 	ListNames	                                   | the owning program                                             |

## dbus工具

qdbusviewer, gui工具，用于查看系统中的所有object, path, inteface, method

dbus-monitor, commandline工具，用于实时显示bus上的request/reply。

## 代码

### python

1. 安装dasdbus
 
```bash
pip3 install dasdbus
```

2. python代码

```python

from dasbus.connection import SessionMessageBus

bus = SessionMessageBus()

proxy = bus.get_proxy("org.gnome.Nautilus", "/org/gnome/Nautilus/SearchProvider", "org.gnome.Shell.SearchProvider2")

res=proxy.GetInitialResultSet(["home"])
print(res)

meta=proxy.GetResultMetas(res)
print(meta)


```

### Qt

Qt有两个工具可以从xml直接生成代码，但是它不是所有的数据类型都直接，不支持的方法可以先删掉然后手动实现。

qdbuscpp2xml和qdbusxml2cpp。



### Gtk

Gtk也有一个工具: https://github.com/Pelagicore/gdbus-codegen-glibmm

## kdbus是啥

kdbus是dbus的`transport layer` + `bus manager`。

bus manager对应的是dbus-deamon, 用kdbus的话，dbus-deamon就不需要了。

[文档](https://www.freedesktop.org/wiki/Software/systemd/kdbus/)
