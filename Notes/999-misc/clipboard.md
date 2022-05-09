# clipboard and selection

在X11系统中，有三种`selection`:

1. PRIMARY
   在终端中左键选择一块区域，区域内的文本就会被存入PRIMARY `selection`。
   当你在其他地方按下鼠标中键的时后，这块文本就会被粘贴进终端。这个是PRIMARY `selection`的常见用过。
2. SECONDARY 没在定义固定的用法，每个应用可以自己定。
3. CLIPBOARD 顾名思义，当你按Ctrl-C或是右键选择`Copy`的时候，选中的文本就会进入CLIPBOARD `selection`。
    粘贴时，就从CLIPBOARD `selection`里拿出本文来。

可以看出，`selection`是对windows下剪贴板的一种范化。

## vim registers

在vim中，`y`和`p`命令，正常操作的不是X11的selection，
因为vim默认认为自己工作于ssh, console等没有X11的环境的。

在vim中，复制和粘贴的中转站叫`registers`，可以用`:help registers`命令查看相关信息：

    
    There are ten types of registers
    1. The unnamed register ""
    2. 10 numbered registers "0 to "9
    3. The small delete register "-
    4. 26 named registers "a to "z or "A to "Z
    5. Three read-only registers ":, "., "%
    6. Alternate buffer register "#
    7. The expression register "=
    8. The selection registers "* and "+
    9. The black hole register "_
    10. Last search pattern register "/

与X11相关的是第8类：

    8. The selection registers "* and "+


在X11环境中，比如Linux Desktop的Gnome-terminal中启动vim，你就可以分别通过`"*p`和`"+p`从
CLIPBOARD `selection`或PRIMARY `selection`中粘贴内容。`y`命令同理。即：

    "*y 和 "*p: 从剪贴板复制粘贴。
    "+y 和 "+p: 从PRIMARY `selection`粘贴。

可以使用`"*`和`"+`的前提是你的vim编译时加了`+clipboard`选项。
查看vim编译时有没有加可以用命令`:echo has('clipboard')`如果返回值是0，说明没有加这个选项。

    **:info:** vimscript中获取寄存器的值的函数是getreg和setreg。可以尝试:echo getreg("*").


## gtk & vte functions

在gtk中，可以通过`gtk_selection_data_set` 和 `gtk_selection_data_set_text`函数设置普通数据和文本到`selection`。
具体参考： http://irtfweb.ifa.hawaii.edu/SoftwareDocs/gtk20/gtk/gtk-selections.html
