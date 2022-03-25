---

title: CMAKE

---


要编译C/C++代码，我们通常使用gcc或g++命令。比如说：

```bash
gcc main.c -o a.out
```

如果你的项目只涉及几个或一、二十个文件，那么只需要写一个脚本去调用gcc就可以了。根本不需要什么构建工具。

但如果你的项目大一点儿，可能事情就没那么简单的了。整个项目被分为多个组件，每个组件都可以独立编译成静态或动态链接库，然后再一起构建最终的可执行文件。更有甚者，你的项目就是一组给别人使用的动态或静态链接库。这时候，如果还是使用一个脚本来构建的话，维护各个组件的依赖关系就变得比较烦人。到这个时候，你面对的实际上有两个问题，一个是用哪些命令编译或链接代码，另一个，怎么维护这些组件间的依赖关系。

# make

make是一个不错的工具，它可以帮我们解决组件间的依赖关系。在makefile中，每个组件都对应一个target、一组dependency和一组commands。

```makefile
a.out: main.c target_b
    gcc main.c -ltargetb -o a.out

target_b: b.c
    gcc b.c -o libtargetb.so
```

make的逻辑足够简单实用，规则也非常灵活，所以它非常强大，你几乎可以用它来构建任何软件。

美中不足的地方就是它太灵活了，以致于在目标被构建出来之前，没有一个工具能够猜测得出这次构建会使用到哪些源代码，编译源代码是会用到哪些参数，带来的后果就是，几乎没有IDE能支持它。

我很喜欢vim，也喜欢命令行敲命令。但是大型软件的开发中，IDE还是比单纯的文本编辑和搜索更有效率（有人可能会argue说vim也可以打造成IDE，但vim打造出来的那个IDE仍然很难或很好的支持makefile)。

# cmake

make十分灵活，它可以用来构建什么东西，不光是C/C++代码，也可以构建其他语言的代码，甚至构建非代码，如文档之类的东西。这也导致了IDE支持它比较难。

cmake限制了这种灵活性，它只支持C/C++代码（当然，它也有一些边角料的命令可以用来做一些其他事情，但这不是cmake的主要部分）。cmake也限制了shell命令的使用，只能使用一些cmake提供的功能。有限的灵活性、严格的语言逻辑，使得cmake对IDE非常友好。目前市面上最好的C/C++ IDE支持cmake。这里所说的最好的IDE指的是Visual Studio和CLion。前者是所有windows程序员都认同的最好的IDE。后者是我认为最好的C/C++ IDE。

*[cmake]: cross platform make

makefile里面的target的生成是由其commands决定了，写下了怎么生成就必须怎么生成，所以你写一条shell语句，就没有办法在windows下编译。与make相比， cmake是**跨平台**的。在CMakeLists.txt中，你不指定一个target怎么编译，你只为它准备材料，主要是说是这个target需要哪些源代码，依赖哪些库。从这个角度上讲，CMakeLists.txt中，程序员只维护target和dependency，而不必管生成target的commands。

那么commands由谁来写呢？cmake帮你写。当你在build文件夹中执行cmake ..时，cmake为你生成Makefile或Visual Studio Project。然后你再调用make或ms build来执行命令，生成target。

跨平台是很好的一件事儿，这意味着你可以使用Visual Studio来写linux下运行的代码，也可以在linux下写Windows上编译的代码。

对于笔者来讲，最大的好处是可以使用Clion这款jetbrains出品的IDE。这样我就可以享受Clion提供的以下服务：
1. 智能改名、提取函数或方法、移动到别的类等高级重构功能
2. 自动补全、自动生成变量名等写代码辅助功能
3. 集成clang-tidy而获得的不良代码提示功能
4. 集成数据库查看器
5. 集成terminal(这里有一个非常好用的、一般terminal没有的功能，一般人我不告诉他)
6. 集成git客户端（这是我用过的最好的git 客户端）
7. 其他一时想不起来或者现在还懒得写的功能

除此之外，如果你不用IDE，cmake至少还有一个优点就是编译过程的打印格式非常的整齐漂亮。

# cmake的故事


>    cmake 是kitware公司以及一些开源开发者在开发几个工具套件(VTK) 的过程中衍生品，最终形成体系，成为一个独立的开放源代码项目。项目的诞生时间是2001年。其官方网站是www.cmake.org，可以通过访问官方网站获得更多关于cmake 的信息。cmake的流行其实要归功于KDE4的开发(似乎跟当年的svn一样，KDE将代码仓库从CVS迁移到SVN，同时证明了SVN管理大型项目的可用性)，在KDE开发者使用了近10年autotools之后，他们终于决定为KDE4选择一个新的工程构建工具，其根本原因用KDE的话来说就是：只有少数几个编译专家能够掌握KDE现在的构建体系(admin/Makefile.common) ，在经历了unsermake, scons 以及cmake 的选型和尝试之后，KDE4决定使用cmake作为自己的构建系统。在迁移过程中，进展异常的顺利，并获得了cmake开发者的支持。所以，目前的KDE4开发版本已经完全使用cmake来进行构建。像kdesvn,rosegarden等项目也开始使用cmake，这也注定了cmake 必然会成为一个主流的构建体系。

以上来自：https://www.kancloud.cn/itfanr/cmake-practice/82983


# 基本语法

## 简单示例
一个基本的CMakeLists.txt是这样的：

```cmake
project (hello)
set(SRC_LIST main.c module1.c module2.c)
add_executable(hello ${SRC_LIST})
```

其中第一句project是可以不用写的。另外，project, set, add_executable等关键字是大小写都可以的。

这里SET是定义一个**变量**，名字为SRC_LIST，值为main.c module1.c module2.c。
add_executable定义了一个**target**，其名字为hello，其类型为executable，即执行文件，${SRC_LIST}是它的源代码输入。

其他定义**target**的方式还有：add_library。

如`add_library(lib1 SHARED lib1.c lib1_module1.c)`
或`add_library(lib2 STATIC lib2.c lib2_module1.c)`

## 链接库及包含子模块

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

## 包含头文件

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

# 全局和局部作用域

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

## 包含来自外部文件夹中的组件

如果我们想包含一个组件，它不在hello的子文件夹里，而是在hello以外，比如说是一个hello的兄弟或堂兄弟文件夹，或包含方式需要有一些变化。如果直接写`add_subdirectory(../lib3)`，则编译时会报错提示你同时指定这个组件的二进制文件生成的路径。因为子文件夹可以很符合逻辑地放在build/lib1, build/lib2下面，但lib3的二进制文件怎么放？build/../lib3，这显示打乱了不扰乱目录的初衷。

放在buid/lib3下也不妥，万一hello下面也有一个子文件夹加lib3呢？

所以这里要求你手动指定：
```cmake
add_subdirectory(../lib3 ${PROJECT_BINARY_DIR}/external/lib3)

```

这里PROJECT_BINARY_DIR是一个CMake的内置变量。默认情况下它的值就是build目录的绝对路径。

## CMake内置变量

上面我们遇到了一个CMake内置变量，其实如果代码写得复杂了，好多CMake内置变量是用得到的。这里只简单介绍常用的，不常用的参考附录。

## 分支控制


## 开关

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

# 基本命令

```cmake
cmake -S . -B build # 指令源代码路径和build路径并configure
cmake --build build # 构建
cmake --build build --target ibus-fun # 构建ibus-fun目标
cmake --build -t test # 构建test目标, 即执行ctest
cpack # 打包
```

其他参数:


```cmake

cmake -B cmake_build_debug -DCMAKE_BUILD_TYPE=Debug # configure为debug构建
cmake --build cmake_build_debug --config Debug # 以Debug的方式构建
ctest -C Debug # 以debug的方式测试
```

跨平台：

```cmake
cmake --build build_unix -G "Unix Makefiles"
cmake --build build_ninja -G "Ninja"
cmake --open <dir> # 打开工程
```


    The following generators are available on this platform (* marks default):
    Green Hills MULTI            = Generates Green Hills MULTI files
                                    (experimental, work-in-progress).
    * Unix Makefiles               = Generates standard UNIX makefiles.
    Ninja                        = Generates build.ninja files.
    Ninja Multi-Config           = Generates build-<Config>.ninja files.
    Watcom WMake                 = Generates Watcom WMake makefiles.
    CodeBlocks - Ninja           = Generates CodeBlocks project files.
    CodeBlocks - Unix Makefiles  = Generates CodeBlocks project files.
    CodeLite - Ninja             = Generates CodeLite project files.
    CodeLite - Unix Makefiles    = Generates CodeLite project files.
    Eclipse CDT4 - Ninja         = Generates Eclipse CDT 4.0 project files.
    Eclipse CDT4 - Unix Makefiles= Generates Eclipse CDT 4.0 project files.
    Kate - Ninja                 = Generates Kate project files.
    Kate - Unix Makefiles        = Generates Kate project files.
    Sublime Text 2 - Ninja       = Generates Sublime Text 2 project files.
    Sublime Text 2 - Unix Makefiles
                                = Generates Sublime Text 2 project files.
