
---

title: rpath

---

# rpath

## 简述

rpath和runpath是可执行文件里带的一个路径，用于运行时搜寻动态链接库。功能上与LD_LIBRARY_PATH一样，就是发生作用的优先级不同，优先级是rpath, LD_LIBRARY_PATH, runpath。

rpath是先出现的，runpath是后出现的。runpath出现为了解决rpath导致的无法用LD_LIBRARY_PATH来override默认库的问题。

## 添加方法
rpath和runpath的使用方法相同，就是在链接时指定：

```
 -Wl,-rpath,${CMAKE_SOURCE_DIR}/thirdparty/install/lib/x86_64-linux-gnu/
```
即rpath逗号后跟一个运行时要寻找动态链接库的路径。

要想使用runpath，要同时指定`-Wl,--enable-new-dtags `，实际上好像编译器已经默认就是了。

## 查看方法

要想查看一个二进制的rpath指定的是啥怎么办？

```bash
 readelf -d ./funterm| head -30

Dynamic section at offset 0x453a0 contains 42 entries:
  标记        类型                         名称/值
 0x0000000000000001 (NEEDED)             共享库：[libvte-2.91.so.0]
 0x0000000000000001 (NEEDED)             共享库：[libgtkmm-3.0.so.1]
 0x0000000000000001 (NEEDED)             共享库：[libatkmm-1.6.so.1]
 0x0000000000000001 (NEEDED)             共享库：[libgdkmm-3.0.so.1]
 0x0000000000000001 (NEEDED)             共享库：[libgtk-3.so.0]
 0x0000000000000001 (NEEDED)             共享库：[libgdk-3.so.0]
 0x0000000000000001 (NEEDED)             共享库：[libgio-2.0.so.0]
 0x0000000000000001 (NEEDED)             共享库：[libglibmm-2.4.so.1]
 0x0000000000000001 (NEEDED)             共享库：[libsigc-2.0.so.0]
 0x0000000000000001 (NEEDED)             共享库：[libpango-1.0.so.0]
 0x0000000000000001 (NEEDED)             共享库：[libgobject-2.0.so.0]
 0x0000000000000001 (NEEDED)             共享库：[libglib-2.0.so.0]
 0x0000000000000001 (NEEDED)             共享库：[libstdc++.so.6]
 0x0000000000000001 (NEEDED)             共享库：[libgcc_s.so.1]
 0x0000000000000001 (NEEDED)             共享库：[libc.so.6]
 0x000000000000001d (RUNPATH)            Library runpath: [/path/to/mybuild/thirdparty/install/lib/x86_64-linux-gnu:]
 0x000000000000000c (INIT)               0x1d000

```

如果是rpath而不是runpath，显示会不一样：

```bash
readelf -d use_shared | grep PATH 
0x0000000f (RPATH)  Library rpath: [./]

```

## 修改方法

我们希望调试时，本地的二进制文件链接是在build目录下的动态链接库，但安装后，链接的是系统目录下的动态链接库。
那就需要在安装前，将二进制文件的rpath修改后再打包到deb里。

```bash
 patchelf --set-rpath '/usr/local/bin/myapp/lib'  myapp
 ```
 rpath可以有多个，用分号分隔。

 patchelf命令没有的话可以用apt install patchelf安装。

 还有一个叫chrpath的命令也可以做到。
 
 ```bash
 chrpath -r"/usr/local/bin/myapp/lib" myapp
 ```


## 调试linker找库的顺序

```bash
LD_DEBUG=libs ./myapp
```

## CMake中的rpath

注意要放在文件头部一点，否则放晚了有可能不生效。

### 设置新构建的二进制文件的rpath

`set(CMAKE_BUILD_RPATH  "asdfaf")`

这里都不检查的，可以随便设，当前运行时就不好用了。


### 设置安装时要修改为的rpath

`set(CMAKE_INSTALL_RPATH "aabb")`

这里都不检查的，可以随便设，当前运行时就不好用了。


### 其他


`set(CMAKE_BUILD_RPATH_USE_ORIGIN True/False)`构建后设置rpath为build下的相关路径。
`set(CMAKE_BUILD_WITH_INSTALL_RPATH True/False)`构建时就用安装时的rpath, 可能动态链接库已经安装好了。

`set(CMAKE_INSTALL_RPATH_USE_LINK_PATH True/False)`在安装时设置二进制的rpath为链接的路径。

`set(CMAKE_SKIP_RPATH TRUE)` 或`set(CMAKE_SKIP_BUILD_RPATH TRUE)`  构建阶段不为二进制加上rpath。
`set(CMAKE_SKIP_INSTALL_RPATH TRUE)` 在安装阶段不但不添加rpath，如果构建结果有的话，还要删掉。


`set(CMAKE_MACOSX_RPATH True)` 在macos下开启。

`set(CMAKE_MSVCIDE_RUN_PATH True)`跟这个话题无关的。。。