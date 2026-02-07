# XDG Base Directory Specification

[链接](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)

该规范只是一个单页文档，定义了一些目录的用途。

$XDG_DATA_HOME: 写入用户特定数据文件, 默认值~/.local/share

$XDG_CONFIG_HOME: 写入用户特定配置文件, 默认值~/.config

$XDG_STATE_HOME: 写入用户特定状态数据, 默认值~/.local/state

无环境变量：写入用户特定可执行文件, 默认值~/.local/bin

$XDG_DATA_DIRS: 数据文件目录集，以`:`分隔。默认值:/usr/local/share:/usr/share。

有一组相对于应该搜索的配置文件的优先顺序基本目录。这组目录由环境变量定义$XDG_CONFIG_DIRS。

有一个相对于应该写入用户特定的非必要（缓存）数据的基本目录。该目录由环境变量定义$XDG_CACHE_HOME。

有一个相对于应该放置用户特定运行时文件和其他文件对象的基本目录。该目录由环境变量定义$XDG_RUNTIME_DIR。

在这些环境变量中设置的所有路径都必须是绝对的。如果实现在任何这些变量中遇到相对路径，它应该认为路径无效并忽略它。