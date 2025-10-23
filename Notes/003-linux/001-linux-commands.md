## [Linux大神都是怎么记住这么多命令的？](https://www.zhihu.com/question/452895041/answer/1818023176)


作者：Nyan Chatora  
链接：https://www.zhihu.com/question/452895041/answer/1818023176  
来源：知乎  
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。  
  

有人说命令很难记

笑死，根本不用记（不是

---

[zsh](https://zhida.zhihu.com/search?content_id=360629211&content_type=Answer&match_order=1&q=zsh&zhida_source=entity)有一些模糊搜索的插件很好用

这些插件大多会利用zsh提供的widget功能，创建自己的ui

比如`history-search-multi-word`，能够搜你自己的历史，这样再难学的命令只要输入一次，就等于是自动记下笔记了

还有`[fzf](https://zhida.zhihu.com/search?content_id=360629211&content_type=Answer&match_order=1&q=fzf&zhida_source=entity)-tab`，这个顾名思义，是用知名的fzf搜索几乎所有可能的自动补全，还会给你一个定制输出内容的机会，比如补全ls的时候同时输出下一级目录内容、补全kill和ps的时候能够补全进程。开始用以后真的是哪里不会tab哪里，毕竟不仅能搜命令参数，还能搜命令参数的描述（我tab键都开始松了）。当然了基本命令怎么用还是得会，而且还有奇葩中的奇葩比如ps和tar这两个魔鬼

那么自动补全从哪儿来？zsh自动补全我写过，要想照顾到方方面面不简单啊

开箱即用做的好的是[fish](https://zhida.zhihu.com/search?content_id=360629211&content_type=Answer&match_order=1&q=fish&zhida_source=entity)，提供了从manpage中读取命令参数的一个Python脚本。无所不包的社区自然是拿了过来，用它来解析manpage输出，然后生成（基本而静态的）bash或zsh自动补全，面对冷门命令即使没有精心写好的自动补全，也差不多够用了

---

剩下的[tldr](https://zhida.zhihu.com/search?content_id=360629211&content_type=Answer&match_order=1&q=tldr&zhida_source=entity)啊fuck啊说的人好像就挺多的了，我并没有用过，听说也挺好用？

---

再说了，常用命令可以自己写包装啊，什么alias都是简单的，再包一层shell函数都很简单，复杂了上Python/Perl/Ruby啊

## 如何在 Linux 中使用 find

> 使用正确的参数，`find` 命令是在你的系统上找到数据的强大而灵活的方式。

在[最近的一篇文章](https://link.zhihu.com/?target=https%3A//linux.cn/article-9585-1.html)中，Lewis Cowles 介绍了 `find` 命令。

`find` 是日常工具箱中功能更强大、更灵活的命令行工具之一，因此值得花费更多的时间。

最简单的，`find` 跟上路径寻找一些东西。例如：

```text
find /
```

它将找到（并打印出）系统中的每个文件。而且由于一切都是文件，你会得到很多需要整理的输出。这可能不能帮助你找到你要找的东西。你可以改变路径参数来缩小范围，但它不会比使用 `ls` 命令更有帮助。所以你需要考虑你想要找的东西。

也许你想在主目录中找到所有的 JPEG 文件。 `-name` 参数允许你将结果限制为与给定模式匹配的文件。

```text
find ~ -name '*jpg'
```

可是等等！如果它们中的一些是大写的扩展名会怎么样？`-iname` 就像 `-name`，但是不区分大小写。

```text
find ~ -iname '*jpg'
```

很好！但是 8.3 名称方案是如此的老。一些图片可能是 .jpeg 扩展名。幸运的是，我们可以将模式用“或”（表示为 `-o`）来组合。

```text
find ~ ( -iname 'jpeg' -o -iname 'jpg' )
```

我们正在接近目标。但是如果你有一些以 jpg 结尾的目录呢？ （为什么你要命名一个 `bucketofjpg` 而不是 `pictures` 的目录就超出了本文的范围。）我们使用 `-type` 参数修改我们的命令来查找文件。

```text
find ~ \( -iname '*jpeg' -o -iname '*jpg' \) -type f
```

或者，也许你想找到那些命名奇怪的目录，以便稍后重命名它们：

```text
find ~ \( -iname '*jpeg' -o -iname '*jpg' \) -type d
```

你最近拍了很多照片，所以让我们把它缩小到上周更改的文件。

```text
find ~ \( -iname '*jpeg' -o -iname '*jpg' \) -type f -mtime -7
```

你可以根据文件状态更改时间 （`ctime`）、修改时间 （`mtime`） 或访问时间 （`atime`） 来执行时间过滤。 这些是在几天内，所以如果你想要更细粒度的控制，你可以表示为在几分钟内（分别是 `cmin`、`mmin`和 `amin`）。 除非你确切地知道你想要的时间，否则你可能会在 `+` （大于）或 `-` （小于）的后面加上数字。

但也许你不关心你的照片。也许你的磁盘空间不够用，所以你想在 `log` 目录下找到所有巨大的（让我们定义为“大于 1GB”）文件：

```text
find /var/log -size +1G
```

或者，也许你想在 `/data` 中找到 bcotton 拥有的所有文件：

```text
find /data -owner bcotton
```

你还可以根据权限查找文件。也许你想在你的主目录中找到对所有人可读的文件，以确保你不会过度分享。

```text
find ~ -perm -o=r
```

这篇文章只说了 `find` 能做什么的表面。将测试条件与[布尔逻辑](https://zhida.zhihu.com/search?content_id=6969945&content_type=Article&match_order=1&q=%E5%B8%83%E5%B0%94%E9%80%BB%E8%BE%91&zhida_source=entity)相结合可以为你提供难以置信的灵活性，以便准确找到要查找的文件。并且像 `-exec` 或 `-delete` 这样的参数，你可以让 `find` 对它发现的内容采取行动。你有任何最喜欢的 `find` 表达式么？在评论中分享它们！

