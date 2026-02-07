---

title: Nautilus扩展开发

---

# 概述

Nautilus是Gnome环境下的文件管理器。

Nautilus支持extension，可以安装一些extension来扩展nautilus的功能。

一般在网上看到的nautilus extension都是javascript或是python开发的。但是由于个人更熟悉C/C++。这里找到了一个C的示例程序。

[C语言Nautilus Extensions Example](https://github.com/hvoigt/simple-nautilus-extension)

上面的链接是github repo，以防万一，下面是打包的代码：
[打包的代码](./assets/simple-nautilus-extension-master.zip)

## 构建及安装

代码里只有一个源文件，源文件只有一百多行，编译为一个.so。
安装只需要把相应的.so复制到指定的extensions目录，即/usr/lib/nautilus/extensions-2.0/。

## 使能

要想使能这个extension，需要重启nautilus.

```bash
killall nautilus; nautilus
```

## 加载路径

`/usr/lib/x86_64-linux-gnu/nautilus/extensions-3.0`

系统里有好多其他组件的extensions加载路径，可以用命令`find /usr -name "extensions*" -type d`查看。

## 编译依赖

使用makefile的话，可以用

`pkg-config --cflags libnautilus-extension`查看包含头文件的路径。
可以用
`pkg-config --libs libnautilus-extension`查看链接的动态链接库。

cmake的话可以用：
```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(libnautilus-extension REQUIRED libnautilus-extension IMPORTED_TARGET)
add_library(xxxx SHARED src/nautilus_extension.cpp)
target_include_directories(xxxx
        PRIVATE ${GTKMM3_INCLUDE_DIRS}
        )
target_link_libraries(xxxx
        ${GTKMM3_LIBRARIES}
        PkgConfig::libnautilus-extension
        )
```        


## 代码解析

这个extension以.so方式存在，nautilus启动时会加载这个so，并在启动和结束时调用两个主要的函数，即：

```c
void nautilus_module_initialize (GTypeModule  *module);
void nautilus_module_shutdown (void);

```
这两个函数需要我们来实现。

在nautilus_module_initialize函数的实现中，我们需要向glib的类型系统注册一个glib的type（这里可以理解为用C语言实现的一个具备反射能力的class,对应java或C++中的类）。这个type就是我们这个extension的具体实现。

nautilus_module_initialize/shutdown有点类似于linux内核的module_init/exit，只是一个模块机制，并没有涉及具体功能。具体功能的提供需要我们实现另一个函数：

```c
void nautilus_module_list_types (const GType **types, int *num_types);
```

nautilus_module_list_types向这个模块查询，你实现了哪些个extension，可以在出台中放置多个。


