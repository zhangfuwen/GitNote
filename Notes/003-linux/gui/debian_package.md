---

title: dpkg and apt command

tags:['dpkg', 'debian', 'apt', 'changelog', 'debchange', 'apt-file','apt-rdepends']

---

# dpkg and apt

# change log

要用debchange来生成changelog

``` bash
sudo apt install devscripts
```

``` bash
debchange "initial commit"
```

# 反向查找依赖该包的其他包

``` bash
if dpkg -s apt-rdepends > /dev/null 2>&1; then
  echo  "apt-rdepends already exists"
else
  sudo apt install apt-rdepends
fi

apt-rdepends -r qtcreator
```

# 查看当前包依赖的包

``` bash
dpkg -s qtcreator
```

# 列出已安装的包

``` bash
apt --installed list
```

# 查找哪些包里安装了这个命令

``` bash
apt-file find ntpdate
```

# 查看某包安装的文件

``` bash
apt-file list ntpdate
```

## dpkg and apt

### 基础操作

```bash
apt update # 更新元数据
apt install xfce4-terminal # 安装
apt remove xfce4-terminal # 删除
apt upgrade # 升级软件包
apt reinstall xfce4-terminal
apt reinstall ~/Downloads/search_xxx.deb
apt source xfce4-terminal # 下载源代码
dpkg -s apt-rdepends # 查询某package是否已安装
apt --installed list # 列出所有已安装的包
2.5.2 依赖查询
2.5.2.1 被依赖查询
if dpkg -s apt-rdepends > /dev/null 2>&1; then
  echo  "apt-rdepends already exists"
else
  sudo apt install apt-rdepends
fi

apt-rdepends -r gedit        
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
gedit
  Reverse Depends: biosyntax-gedit (1.0.0b-2)
  Reverse Depends: cinnamon-desktop-environment (5.2.2)
  Reverse Depends: gedit-dev (<< 41.0-3)
  Reverse Depends: gedit-latex-plugin (>= 3.20.0-2)
  Reverse Depends: gedit-plugin-bookmarks (<< 41.0-1)
  Reverse Depends: gedit-plugin-bracket-completion (>= 41.0-1)
  Reverse Depends: gedit-plugin-character-map (>= 41.0-1)
  Reverse Depends: gedit-plugin-code-comment (>= 41.0-1)
  Reverse Depends: gedit-plugin-color-picker (>= 41.0-1)
  Reverse Depends: gedit-plugin-color-schemer (>= 41.0-1)
  Reverse Depends: gedit-plugin-commander (>= 41.0-1)
  Reverse Depends: gedit-plugin-draw-spaces (>= 41.0-1)
  Reverse Depends: gedit-plugin-find-in-files (>= 41.0-1)
  Reverse Depends: gedit-plugin-git (>= 41.0-1)
  Reverse Depends: gedit-plugin-join-lines (>= 41.0-1)
  Reverse Depends: gedit-plugin-multi-edit (>= 41.0-1)
  Reverse Depends: gedit-plugin-session-saver (>= 41.0-1)
  Reverse Depends: gedit-plugin-smart-spaces (>= 41.0-1)
  Reverse Depends: gedit-plugin-synctex (>= 41.0-1)
  Reverse Depends: gedit-plugin-terminal (>= 41.0-1)
  Reverse Depends: gedit-plugin-text-size (>= 41.0-1)
  Reverse Depends: gedit-plugin-translate (>= 41.0-1)
  Reverse Depends: gedit-plugin-word-completion (>= 41.0-1)
  Reverse Depends: gedit-source-code-browser-plugin (>= 3.0.3-6)
  Reverse Depends: gnome-core (>= 1:42+3)
  Reverse Depends: rabbitvcs-gedit (0.18-3)
  Reverse Depends: supercollider-gedit (>= 1:3.11.2+repack-1build1)
  Reverse Depends: task-gnome-flashback-desktop (3.68ubuntu2)
  Reverse Depends: ubuntu-unity-desktop (0.4)
biosyntax-gedit
  Reverse Depends: biosyntax (1.0.0b-2)
biosyntax
cinnamon-desktop-environment
  Reverse Depends: task-cinnamon-desktop (3.68ubuntu2)
task-cinnamon-desktop
gedit-dev
gedit-latex-plugin
gedit-plugin-bookmarks
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugins
gedit-plugin-bracket-completion
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-character-map
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-code-comment
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-color-picker
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-color-schemer
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-commander
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-draw-spaces
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-find-in-files
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-git
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-join-lines
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-multi-edit
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-session-saver
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-smart-spaces
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-synctex
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-terminal
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-text-size
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-translate
  Reverse Depends: gedit-plugins (41.0-1)
gedit-plugin-word-completion
  Reverse Depends: gedit-plugins (41.0-1)
gedit-source-code-browser-plugin
gnome-core
  Reverse Depends: gnome (= 1:42+3)
  Reverse Depends: task-gnome-desktop (3.68ubuntu2)
gnome
task-gnome-desktop
rabbitvcs-gedit
supercollider-gedit
task-gnome-flashback-desktop
ubuntu-unity-desktop
```


#### 依赖查询

```
 apt depends gedit
gedit
  依赖: gedit-common (<< 42)
  依赖: gedit-common (>= 41)
  依赖: gir1.2-glib-2.0
  依赖: gir1.2-gtk-3.0 (>= 3.22)
  依赖: gir1.2-gtksource-4
  依赖: gir1.2-pango-1.0
  依赖: gir1.2-peas-1.0
  依赖: gsettings-desktop-schemas
  依赖: iso-codes
  依赖: python3-gi (>= 3.0)
  依赖: python3-gi-cairo (>= 3.0)
  依赖: <python3:any>
    python3:i386
    python3
  依赖: python3.10
  依赖: libatk1.0-0 (>= 1.12.4)
  依赖: libc6 (>= 2.34)
  依赖: libcairo2 (>= 1.2.4)
  依赖: libgdk-pixbuf-2.0-0 (>= 2.22.0)
  依赖: libgirepository-1.0-1 (>= 0.9.3)
  依赖: libglib2.0-0 (>= 2.64)
  依赖: libgspell-1-2 (>= 1.8.2)
  依赖: libgtk-3-0 (>= 3.22)
  依赖: libgtksourceview-4-0 (>= 3.18.0)
  依赖: libpango-1.0-0 (>= 1.42.0)
  依赖: libpeas-1.0-0 (>= 1.14.1)
  依赖: libxml2 (>= 2.7.4)
  推荐: yelp
  推荐: zenity
  建议: gedit-plugins

  
```  

### 探索一个package
#### remote package 描述信息

```
➜  ~ apt show search
 apt show gedit
Package: gedit
Version: 41.0-3
Priority: optional
Section: gnome
Origin: Ubuntu
Maintainer: Ubuntu Developers <ubuntu-devel-discuss@lists.ubuntu.com>
Original-Maintainer: Debian GNOME Maintainers <pkg-gnome-maintainers@lists.alioth.debian.org>
Bugs: https://bugs.launchpad.net/ubuntu/+filebug
Installed-Size: 1,827 kB
Depends: gedit-common (<< 42), gedit-common (>= 41), gir1.2-glib-2.0, gir1.2-gtk-3.0 (>= 3.22), gir1.2-gtksource-4, gir1.2-pango-1.0, gir1.2-peas-1.0, gsettings-desktop-schemas, iso-codes, python3-gi (>= 3.0), python3-gi-cairo (>= 3.0), python3:any, python3.10, libatk1.0-0 (>= 1.12.4), libc6 (>= 2.34), libcairo2 (>= 1.2.4), libgdk-pixbuf-2.0-0 (>= 2.22.0), libgirepository-1.0-1 (>= 0.9.3), libglib2.0-0 (>= 2.64), libgspell-1-2 (>= 1.8.2), libgtk-3-0 (>= 3.22), libgtksourceview-4-0 (>= 3.18.0), libpango-1.0-0 (>= 1.42.0), libpeas-1.0-0 (>= 1.14.1), libxml2 (>= 2.7.4)
Recommends: yelp, zenity
Suggests: gedit-plugins
Homepage: https://wiki.gnome.org/Apps/Gedit
Task: ubuntu-desktop-minimal, ubuntu-desktop, ubuntu-budgie-desktop, ubuntu-budgie-desktop-raspi
Download-Size: 434 kB
APT-Manual-Installed: yes
APT-Sources: http://cn.archive.ubuntu.com/ubuntu jammy/main amd64 Packages
Description: GNOME 桌面环境的官方文本编辑器
 gedit 是一个文本编辑器，支持大多数标准编辑器的功能，同时还包含有一些在其 他普通文本编辑器中没有的功能。gedit
 是一个图形界面的程序，支持在一个窗口 内编辑多个文本文件(有时被称为标签或 MDI)。
 .
 gedit 使用 Unicode 的 UTF-8 来编码文件，故能够支持各种国际语言的文本。
 它的核心功能包括支持源代码的语法高亮、自动缩进和打印及打印预览支持。
 .
 gedit 可以通过它的插件系统来扩展功能，目前包括支持拼写检查、文件比较、查看 CVS 更新日志以及调整缩进层次。
```

#### installed package所含文件：

```
dpkg-query -L gedit | head
/.
/usr
/usr/bin
/usr/bin/gedit
/usr/lib
/usr/lib/python3
/usr/lib/python3/dist-packages
/usr/lib/python3/dist-packages/gi
/usr/lib/python3/dist-packages/gi/overrides
/usr/lib/python3/dist-packages/gi/overrides/Gedit.py
```

#### deb包所含文件

```
dpkg-deb --contents ~/Downloads/python3-multibootusb_9.2.0-1_all.deb | head
drwxr-xr-x root/root         0 2018-04-05 01:14 ./
drwxr-xr-x root/root         0 2018-04-05 01:14 ./usr/
drwxr-xr-x root/root         0 2018-04-05 01:14 ./usr/bin/
-rwxr-xr-x root/root      7513 2018-04-05 01:14 ./usr/bin/multibootusb
-rwxr-xr-x root/root       614 2017-10-25 02:43 ./usr/bin/multibootusb-pkexec
drwxr-xr-x root/root         0 2018-04-05 01:14 ./usr/lib/
drwxr-xr-x root/root         0 2018-04-05 01:14 ./usr/lib/python3/
drwxr-xr-x root/root         0 2018-04-05 01:14 ./usr/lib/python3/dist-packages/
-rw-r--r-- root/root       437 2018-04-05 01:14 ./usr/lib/python3/dist-packages/multibootusb-9.2.0.egg-info
drwxr-xr-x root/root         0 2018-04-05 01:14 ./usr/lib/python3/dist-packages/scripts/

```

#### remote package所含文件

```
➜  ~ apt --installed list | grep xfce4-terminal

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

➜  ~ apt-file list xfce4-terminal | head       
xfce4-terminal: /usr/bin/xfce4-terminal
xfce4-terminal: /usr/bin/xfce4-terminal.wrapper
xfce4-terminal: /usr/share/applications/xfce4-terminal-settings.desktop
xfce4-terminal: /usr/share/applications/xfce4-terminal.desktop
xfce4-terminal: /usr/share/doc/xfce4-terminal/AUTHORS
xfce4-terminal: /usr/share/doc/xfce4-terminal/HACKING
xfce4-terminal: /usr/share/doc/xfce4-terminal/NEWS.Debian.gz
xfce4-terminal: /usr/share/doc/xfce4-terminal/NEWS.gz
xfce4-terminal: /usr/share/doc/xfce4-terminal/README.Debian
xfce4-terminal: /usr/share/doc/xfce4-terminal/README.md
```

## 文件属于哪个package

```
dpkg-query --search /usr/bin/grub-file
grub-common: /usr/bin/grub-file
```
