---

title: dbus开发调试工具和方法

---

# 概述

dbus是linux下提供跨进程服务的一套机制，类以于Android下的binder。不同的地dbus一般是基于unix domain socket或是Shared Memory Transport, named pipe等。

无论dbus底下是啥，它的效率是不如binder的。

## dbus底层的transport是什么

UDS, SMT, TCP?

## kdbus是啥

