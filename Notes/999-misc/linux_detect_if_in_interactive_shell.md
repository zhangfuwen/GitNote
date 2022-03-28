---

title: Linux下判断当前运行环境是不是一个interactive shell

---

我在写一个命令行小程序，我想在interactive shell中，按top的样式刷新屏幕，而如果输出被重定向到文件，则去掉各类颜色、清屏、\r之类的输出。
所以我搜了一下怎么判断当前是不是处于一个interactive shell中的方法，在stackoverflow中有好多方法，这里记录一下.

https://unix.stackexchange.com/questions/26676/how-to-check-if-a-shell-is-login-interactive-batch

# 通过tty判断

```bash
if tty -s
then
    echo Terminal
else
    echo Not on a terminal
fi
```

或

```bash
if [ "`tty`" != "not a tty" ]; then
    echo Terminal
fi
```

# 通过$-判断

```bash
case $- in
  *i*) echo "This shell is interactive";;
  *) echo "This is a script";;
esac
```

而什么是$-呢？

根据[这个链接](https://stackoverflow.com/questions/42757236/what-does-mean-in-bash):

    $- prints The current set of options in your current shell.

    himBH means following options are enabled:

    H - histexpand
    m - monitor
    h - hashall
    B - braceexpand
    i - interactive


# 如何在C/C++代码里判断？

## C语言中获取环境变量的办法

```c
#include <stdlib.h>
#include <stdio.h>

int main(void)
{
    char *pathvar;
    pathvar = getenv("PATH");
    printf("pathvar=%s",pathvar);
    return 0;
}

```

然而`$-`不是一个环境变量。你可以用SHELL这个环境变量。

## istty

    NAME         top
        isatty - test whether a file descriptor refers to a terminal
    SYNOPSIS         top
        #include <unistd.h>

        int isatty(int fd);
    DESCRIPTION         top
        The isatty() function tests whether fd is an open file descriptor
        referring to a terminal.
    RETURN VALUE         top
        isatty() returns 1 if fd is an open file descriptor referring to
        a terminal; otherwise 0 is returned, and errno is set to indicate
        the error.

```c
#include <unistd.h>
int is_redirected(){
   if (!isatty(fileno(stdout))){
       fprintf(stdout, "argv, argc, someone is redirecting me elsewhere...\n");
       return 1;
   }
   return 0;
}

```