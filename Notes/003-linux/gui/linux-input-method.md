# Linux 输入法概述

# 1\. 概述

Linux 有三个主要的输入法接口：

* XIM ( X Input Method, X 协议提供的输入法接口）
* GTK IM module
* QT IM module

所认Linux用户在设置输入法时，经常要设置三个环境变量：

```
export XMODIFIERS=@im=fcitx
export QT_IM_MODULE=fcitx
export GTK_IM_MODULE=fcitx
```

输入法将键盘的按钮事件转换成unicode字符，这个三套API定义了application获取输入法转换后的unicode字符的方式。不同的应用使用不同的接口：

1. 有些应用直接调用libx或xcb的API来产生窗口并接受用户输入，这时它只能用XIM提供的接口。
2. 有些应用是使用Qt开发的，它天然使用是QT IM MODULE提供的输入法接口
3. 有些应用是使用GTK开发的，它就使用GTK IM MODULE提供的输入法接口

# 2\. XIM使用示例

以下代码引自：
A Brief Intro to Input Method Framework, Linux IME, and XIM
tedyin.com/posts/a-brief-intro-to-linux-input-method-framework/

![](/assets/res/2021-11-03-17-18-58.png)

从代码中可以看到，应用直接调用X开头的函数(X protocol函数）来创建窗口，它通过调用XSetLocaleModifieers函数来设置要使用的输入法引擎，22行设置输入法引擎为fcitx。代码里是这样写，但真实的应该不会设置一个具体的输入法引擎，而是设置成空字符串，这意味着应用不自已选输入法引擎，而是听从环境变量XMODIFIERS的指示。

还有一些其他设置要做，这里不解释，大家可以看原始链接。

应用初始化完输入法引擎之后，如何获取输入呢？通过Xutf8LookupString，并传入按键的值。

![](/assets/res/2021-11-03-17-22-39.png)

并非每次传入一个按钮值输入法都给出一个输出，因为返回值c有时是0。不为零时，才会得到一个unicode字符：

![](/assets/res/2021-11-03-17-22-57.png)

# 3\. 输入法框架

XIM, GTK IM Module, QT IM Module称为输入法接口（有人也叫框架，这里没有固定说法，我们叫接口吧）。我们常用的输入法ibus, fcitx等称为输入法框架（Framework)。也有人把他们也叫二级框架。为了表述一致，我们还是称为输入法框架。

输入法框架历史上有过好多：

https://blogs.gnome.org/happyaron/2011/01/15/linux-input-method-brief-summary/
blogs.gnome.org/happyaron/2011/01/15/linux-input-method-brief-summary/
现今最常用的是ibus和fcitx。ibus是多数以gnome为默认桌面环境的发行版本默认输入法。fcitx则是中国人最喜欢的输入法框架，因为它一开始是专为中文输入开发的，它的全称是Free Chinese Input Toy for X。ibus的全称是Intelligent Input Bus。

输入法框架本身不提供具体的按键到unicode的映射，它需要输入法引擎来提供具体功能。比如说ibus这个输入法框架下面有如下输入法引擎，或插件：（引自https://en.wikipedia.org/wiki/Intelligent\_Input\_Bus）

```
ibus-anthy: A plugin for Anthy, a Japanese IME.
ibus-avro: Phonetic keyboard layout for writing Bengali based on Avro Keyboard[8][9][10]
ibus-cangjie:[11] An engine for the Cangjie input method.
ibus-chewing: An intelligent Chinese Phonetic IME for Zhùyīn users. It is based on libChewing.
ibus-hangul: A Korean IME.
ibus-libpinyin: A newer Chinese IME for Pinyin users. Designed by Huang Peng and Peng Wu.
ibus-libthai: A Thai IME based on libthai.
ibus-libzhuyin:[12] An engine for the Zhùyīn ("bopomofo") input method (an alternative to ibus-chewing).
ibus-m17n: A m17n IME which allows input of many languages using the input methods from m17n-db. See more details in #ibus-m17n.
ibus-mozc: A plugin to the Japanese IME "mozc" developed by Google.[13]
ibus-pinyin: An intelligent Chinese Phonetic IME for Hanyu pinyin users. Designed by Huang Peng (main author of IBus) and has many advanced features such as English spell checking.
ibus-table: An IME that accommodates table-based IMs. See more details in #ibus-table.
ibus-unikey: An IME for typing Vietnamese characters.
```

从里面可以看来，输入法引擎的原理各不相同。有的引擎是具有复杂算法的，比如说常词、网络词库之类的。有的输入法引擎超级简单，它就是在查字典。所以查字典的引擎都没有必要单独搞一个引擎，而是直接使用ibus-table引擎，再配合不同的字典：

```
Officially released IM tables:[15]

latex: Input special characters using LaTeX syntax. Included in ibus-table package.
compose: input special letter by compose letter and diacritical mark. Included in ibus-table package.
Array30: Array30 Chinese IM tables.
Cangjie: Cangjie 3 and 5 Chinese IM tables.
Erbi: Er-bi Chinese IM table.
Wubi: Wubi Chinese IM table.
Yong: YongMa Chinese IM Table.
ZhengMa: ZhengMa Chinese IM table.
```

比如说ubuntu有一个软件包叫做ibus-table-wubi，就是一个字典：

[acevery/ibus-table-wubi](https://link.zhihu.com/?target=https%3A//github.com/acevery/ibus-table-wubi)

其内容大概是这样的：

![](/assets/res/2021-11-03-17-26-03.png)

# 4\. 软件栈\(ibus\)

引自：https://www.chromium.org/chromium-os/chromiumos-design-docs/text-input

![](/assets/res/2021-11-03-17-26-22.png)

上图是一个ibus + GTK IM Module + Chromium的软件栈示意图。

当用户敲击一个按键时产生一个key code，按键的key symbol通过X protocol的XKB扩展传给Chromium进程，Chromium进程实际接收这个事件的组件是GTK IM Module。GTK IM Module从环境变量中知道当前使用的输入法是ibus后，就会加载im-ibus.so这个动态链接库，并把symbol传给它。im-ibus.so的被调用函数通过D-Bus与ibus-daemon通信，ibus-daemon再通过D-Bus与具体的输入法引擎通信获取到unicode字符，图中可以是ibus-pinyin。

下面这张图也能说明一些问题：[sil2100//vx home page](https://link.zhihu.com/?target=http%3A//sil2100.vexillium.org/%3Fid%3Ddev%26sid%3D33)

![](/assets/res/2021-11-03-17-26-36.png)

下面这张图列出了所有三个接口情况下Ibus的架构：https://www.slideshare.net/ftake/what-is-necessary-for-the-next-input-method-framework

![](/assets/res/2021-11-03-17-27-34.png)

实际上Fcitx也是类似的，所有的IMF(输入法框架）都是类似的架构：

![](/assets/res/2021-11-03-17-27-44.png)

1. 应用输入不了中文如何深度定位？
2. 检查输入法在其他应用是否能输入
3. 换一个输入法框架看看能否输入，如ibus换成fcitx
4. 查看ibus-deamon是否存在
5. 如果是gtk程序比如说(chromium)，检查程序运行的环境变量中GTK\_IM\_MODULE 环境变量是否正确设置了。查看进程的环境变量 - wanlifeipeng - 博客园
6. 查看进程是否加载了im-ibus.so，lsof -p 进程id，这个命令是查看进程打开的文件，进程加载的动态链接库也是进程打开的文件。
7. gdb调试，看相应的函数有没有被调用, 客户端im-ibus.so的代码：ibus/ibus
8. DBUS监听，看看应用有没有通过dbus给ibus-daemon发送数据