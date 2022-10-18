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

apt-rdepends -r ocean-framework-qt


#### 依赖查询
➜  ~ apt depends search
search
  依赖: qml-module-qtquick-controls2
  依赖: qml-module-qtquick2
  依赖: qml-module-qtquick-layouts
  依赖: qml-module-qt-labs-platform
  依赖: qml-module-qt-labs-settings
  依赖: qml-module-qtqml
  依赖: qml-module-qtquick-window2
  依赖: qml-module-qtquick-shapes
  依赖: qml-module-qtgraphicaleffects
  依赖: kwin-wayland
  依赖: kwin-wayland-backend-drm
  依赖: kwin-wayland-backend-wayland
  依赖: kwin-wayland-backend-x11
  依赖: kwin-wayland-backend-fbdev
  依赖: kwin-wayland-backend-virtual
  依赖: breeze-icon-theme
  依赖: ocean-framework-qt (>= 1.0+git20220801.939)
  依赖: kio
  依赖: libc6 (>= 2.14)
  依赖: libgcc-s1 (>= 3.4)
  依赖: libkf5baloo5 (>= 5.3.0+git20150512)
  依赖: libkf5kiocore5 (>= 5.69.0)
  依赖: libqt5core5a (>= 5.15.1)
  依赖: libqt5dbus5 (>= 5.14.1)
 |依赖: libqt5gui5 (>= 5.7.0)
  依赖: libqt5gui5-gles (>= 5.7.0)
  依赖: libqt5qml5 (>= 5.1.0)
 |依赖: libqt5quick5 (>= 5.0.2)
  依赖: libqt5quick5-gles (>= 5.0.2)
  依赖: libqt5widgets5 (>= 5.0.2)
  依赖: libstdc++6 (>= 9)

### 探索一个package
#### remote package 描述信息
➜  ~ apt show search
Package: search
Version: 0.1+git20220803.1009-0
Status: install ok installed
Priority: optional
Section: devel
Maintainer: zhangfuwen <zhangfuwen@bytedance.com>
Installed-Size: 276 kB
Depends: qml-module-qtquick-controls2, qml-module-qtquick2, qml-module-qtquick-layouts, qml-module-qt-labs-platform, qml-module-qt-labs-settings, qml-module-qtqml, qml-module-qtquick-window2, qml-module-qtquick-shapes, qml-module-qtgraphicaleffects, kwin-wayland, kwin-wayland-backend-drm, kwin-wayland-backend-wayland, kwin-wayland-backend-x11, kwin-wayland-backend-fbdev, kwin-wayland-backend-virtual, breeze-icon-theme, ocean-framework-qt (>= 1.0+git20220801.939), kio, libc6 (>= 2.14), libgcc-s1 (>= 3.4), libkf5baloo5 (>= 5.3.0+git20150512), libkf5kiocore5 (>= 5.69.0), libqt5core5a (>= 5.15.1), libqt5dbus5 (>= 5.14.1), libqt5gui5 (>= 5.7.0) | libqt5gui5-gles (>= 5.7.0), libqt5qml5 (>= 5.1.0), libqt5quick5 (>= 5.0.2) | libqt5quick5-gles (>= 5.0.2), libqt5widgets5 (>= 5.0.2), libstdc++6 (>= 9)
Homepage: https://www.bytedance.com/
Download-Size: 未知
APT-Manual-Installed: yes
APT-Sources: /var/lib/dpkg/status
Description: Ocean search tool

#### installed package所含文件：
➜  ~ dpkg-query -L ocean-framework-qt
/.
/usr
/usr/lib
/usr/lib/libnetworksetting.so
/usr/lib/x86_64-linux-gnu
/usr/lib/x86_64-linux-gnu/libpackagemanager-qt.so.1.0
/usr/lib/x86_64-linux-gnu/libwallpapermanager-qt.so.1.0
/usr/lib/x86_64-linux-gnu/qt5
/usr/lib/x86_64-linux-gnu/qt5/qml
/usr/lib/x86_64-linux-gnu/qt5/qml/ocean
/usr/lib/x86_64-linux-gnu/qt5/qml/ocean/audio

#### deb包所含文件
➜  ~ dpkg-deb -c ~/Downloads/settings_1.0.0_amd64.deb | head
drwxr-xr-x root/root         0 2022-06-09 21:37 ./
drwxr-xr-x root/root         0 2022-06-09 21:37 ./opt/
drwxr-xr-x root/root         0 2022-06-09 21:37 ./opt/user-settings/
drwxr-xr-x root/root         0 2022-06-09 21:37 ./opt/user-settings/Applet/
drwxr-xr-x root/root         0 2022-06-09 21:37 ./opt/user-settings/Applet/AboutApplet/
-rw-r--r-- root/root      2280 2022-06-09 21:37 ./opt/user-settings/Applet/AboutApplet/icon_about.svg
drwxr-xr-x root/root         0 2022-06-09 21:37 ./opt/user-settings/Applet/AboutApplet/image/
-rw-r--r-- root/root       368 2022-06-09 21:37 ./opt/user-settings/Applet/AboutApplet/image/arrow_left.png
-rw-r--r-- root/root       406 2022-06-09 21:37 ./opt/user-settings/Applet/AboutApplet/image/arrow_left_dark.png
-rw-r--r-- root/root       313 2022-06-09 21:37 ./opt/user-settings/Applet/AboutApplet/image/arrow_right.png

#### remote package所含文件
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

## 文件属于哪个package

➜  ~ dpkg-query -S /usr/bin/Search
search: /usr/bin/Search
