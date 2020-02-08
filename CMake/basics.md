# CMake基本原理和常用技巧

## 基本原理

### 简单示例
一个基本的CMakeLists.txt是这样的：

```cmake
PROJECT (HELLO)
SET(SRC_LIST main.c module1.c module2.c)
ADD_EXECUTABLE(hello ${SRC_LIST})
```

其中第一句project是可以不用写的。另外，project, set, add_executable等关键字是大小写都可以的。

这里SET是定义一个**变量**，名字为SRC_LIST，值为main.c module1.c module2.c。
add_executable定义了一个**target**，其名字为hello，其类型为executable，即执行文件，${SRC_LIST}是它的源代码输入。

其他定义**target**的方式还有：add_library。

如`add_library(lib1 SHARED lib1.c lib1_module1.c)`
或`add_library(lib2 STATIC lib2.c lib2_module1.c)`

### 链接库及包含子模块

假设lib1文件夹下有一个动态链接库， lib2文件夹下有一个静态连接库，我们的hello组件怎么使用他们？


```cmake
PROJECT (HELLO)
SET(SRC_LIST main.c module1.c module2.c)
ADD_EXECUTABLE(hello ${SRC_LIST})

add_subdirectory(lib1)
add_subdirectory(lib2)
target_link_libraries(hello lib1 lib2)
```

add_subdirectory相当于C或是makefile中的include，这要求lib1, lib2文件夹下各自要有一个CMakeLists.txt。而且要定义了一个动态连接库的target和一个静态链接库的target。

lib1下的CMakeLists.txt:

```cmake
set(Sources lib1.c lib1_module1.c)
add_library(lib1 SHARED ${Sources})
```

lib2下的CMakeLists.txt:
```cmake
add_library(lib2 STATIC lib2.c lib2_module1.c)
```

这里lib2换了一种写法，lib1的写法，即先把源代码列表定义为一个变量，然后以${}去引用，这种方式更好一些。但文件较少的话，也可以采用lib2的写法。

### 包含头文件

如果lib1中有一个lib1.h，在main.c中需要包含，我们可以在main.c中写为

```cmake
#include "lib1/lib1.h"
```

但有时候我们想写基于lib1的目录的相对路径，比如想包含的是lib1/api/api.h，我们希望在main.c中写

```cmake
#include "api/api.h"
```

这时我们希望在hello中加一个头文件搜索路径，即在gcc中加一条`-Ilib1`，代码应该这样写：

```cmake
target_include_directories(hello lib1)
```

全文为：

```cmake
PROJECT (HELLO)
SET(SRC_LIST main.c module1.c module2.c)
ADD_EXECUTABLE(hello ${SRC_LIST})

add_subdirectory(lib1)
add_subdirectory(lib2)
target_link_libraries(hello lib1 lib2)
target_include_directories(hello lib1)

```

## 全局和局部作用域

target_link_libraries和target_include_directories两条指令都只作用于hello这个**target**，如果你在当前CMakeLists.txt中还定义了别的**target**，则是不起作用的。

```cmake
PROJECT (HELLO)
SET(SRC_LIST main.c module1.c module2.c)
ADD_EXECUTABLE(hello ${SRC_LIST})
add_executable(hello2 main2.c)

add_subdirectory(lib1)
add_subdirectory(lib2)
target_link_libraries(hello lib1 lib2)
target_include_directories(hello lib1)

```

上述代码中，增加了一个名为hello2的**target**，它的编译输入main2.c，但是在main2.c中不能直接`include "api/api.h"`, 必须`include "lib1/api/api.h"`，同理hello2也不能链接使用lib1和lib2中的符号。

target_link_libraries和target_include_directories是一个局域作用域的指令，它们的作用域限定为指定的**target**。它们还有一个对应的全局作用域版本，为link_libraries和include_directories。全局作用域版本对于这两条指令之后的所有target都生效，对之后包含的子目录中定义的target也生效。语法就是不用写**target**。

```cmake
PROJECT (HELLO)

# dependencies
add_subdirectory(lib1)
add_subdirectory(lib2)
link_libraries(hello lib1 lib2)
include_directories(hello lib1)

# targets
SET(SRC_LIST main.c module1.c module2.c)
ADD_EXECUTABLE(hello ${SRC_LIST})
add_executable(hello2 main2.c)

```

### 包含来自外部文件夹中的组件

如果我们想包含一个组件，它不在hello的子文件夹里，而是在hello以外，比如说是一个hello的兄弟或堂兄弟文件夹，或包含方式需要有一些变化。如果直接写`add_subdirectory(../lib3)`，则编译时会报错提示你同时指定这个组件的二进制文件生成的路径。因为子文件夹可以很符合逻辑地放在build/lib1, build/lib2下面，但lib3的二进制文件怎么放？build/../lib3，这显示打乱了不扰乱目录的初衷。

放在buid/lib3下也不妥，万一hello下面也有一个子文件夹加lib3呢？

所以这里要求你手动指定：
```cmake
add_subdirectory(../lib3 ${PROJECT_BINARY_DIR}/external/lib3)

```

这里PROJECT_BINARY_DIR是一个CMake的内置变量。默认情况下它的值就是build目录的绝对路径。

### CMake内置变量

上面我们遇到了一个CMake内置变量，其实如果代码写得复杂了，好多CMake内置变量是用得到的。这里只简单介绍常用的，不常用的参考附录。

### 分支控制


### 开关

1. 字符串判断
```cmake
SET(CMAKE_BUILD_TYPE Debug)
IF (CMAKE_BUILD_TYPE STREQUAL Debug)        
    ADD_DEFINITIONS(-DDEBUG)                 
ENDIF(CMAKE_BUILD_TYPE STREQUAL Debug)
```

2. ON/OFF
```cmake
SET(ENABLE_DEBUG ON)
IF(ENABLE_DEBUG)
    ADD_DEFINITIONS(-DDEBUG)
ENDIF(ENABLE_DEBUG)
```