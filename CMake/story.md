# 前言

要编译C/C++代码，我们通常使用gcc或g++命令。比如说：

```bash
gcc main.c -o a.out
```

如果你的项目只涉及几个或一、二十个文件，那么只需要写一个脚本去调用gcc就可以了。根本不需要什么构建工具。

但如果你的项目大一点儿，可能事情就没那么简单的了。整个项目被分为多个组件，每个组件都可以独立编译成静态或动态链接库，然后再一起构建最终的可执行文件。更有甚者，你的项目就是一组给别人使用的动态或静态链接库。这时候，如果还是使用一个脚本来构建的话，维护各个组件的依赖关系就变得比较烦人。到这个时候，你面对的实际上有两个问题，一个是用哪些命令编译或链接代码，另一个，怎么维护这些组件间的依赖关系。

## make

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

## cmake

make十分灵活，它可以用来构建什么东西，不光是C/C++代码，也可以构建其他语言的代码，甚至构建非代码，如文档之类的东西。这也导致了IDE支持它比较难。

cmake限制了这种灵活性，它只支持C/C++代码（当然，它也有一些边角料的命令可以用来做一些其他事情，但这不是cmake的主要部分）。cmake也限制了shell命令的使用，只能使用一些cmake提供的功能。有限的灵活性、严格的语言逻辑，使得cmake对IDE非常友好。目前市面上最好的C/C++ IDE支持cmake。这里所说的最好的IDE指的是Visual Studio和CLion。前者是所有windows程序员都认同的最好的IDE。后者是我认为最好的C/C++ IDE。

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

## cmake的故事

    cmake 是kitware公司以及一些开源开发者在开发几个工具套件(VTK) 的过程中衍生品，最终形成体系，成为一个独立的开放源代码项目。项目的诞生时间是2001年。其官方网站是www.cmake.org，可以通过访问官方网站获得更多关于cmake 的信息。cmake的流行其实要归功于KDE4的开发(似乎跟当年的svn一样，KDE将代码仓库从CVS迁移到SVN，同时证明了SVN管理大型项目的可用性)，在KDE开发者使用了近10年autotools之后，他们终于决定为KDE4选择一个新的工程构建工具，其根本原因用KDE的话来说就是：只有少数几个编译专家能够掌握KDE现在的构建体系(admin/Makefile.common) ，在经历了unsermake, scons 以及cmake 的选型和尝试之后，KDE4决定使用cmake作为自己的构建系统。在迁移过程中，进展异常的顺利，并获得了cmake开发者的支持。所以，目前的KDE4开发版本已经完全使用cmake来进行构建。像kdesvn,rosegarden等项目也开始使用cmake，这也注定了cmake 必然会成为一个主流的构建体系。

以上来自：https://www.kancloud.cn/itfanr/cmake-practice/82983

