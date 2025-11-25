---

title: linux app/service配置

---


# desktop 文件

[标准文档](https://specifications.freedesktop.org/desktop-entry-spec/latest/ar01s06.html)

```
[Desktop Entry]
Name=Tracker File System Miner
Comment=Crawls and processes files on the file system
Exec=@libexecdir@/tracker-miner-fs-3
Terminal=false
Type=Application
Categories=Utility;
X-GNOME-Autostart-enabled=true
X-GNOME-HiddenUnderSystemd=true
X-KDE-autostart-after=panel
X-KDE-StartupNotify=false
X-KDE-UniqueApplet=true
NoDisplay=true
OnlyShowIn=GNOME;KDE;XFCE;X-IVI;Unity;
X-systemd-skip=true
```



桌面入口规范为应用程序启动器创建了一个标准。Gnome 为广泛使用的格式添加了几个扩展，但据我所知，没有记录。这是尝试记录它们，以便我可以为 gnome 编写自己的自动启动启动器。非常欢迎拉取请求。

有一个关于 gnome 开发人员的指南，它解释了有关如何将应用程序与桌面集成的基础知识。

自动启动应用程序在用户登录到图形桌面环境时运行。所有桌面管理器都会对格式进行自定义扩展。这只是为了涵盖 Gnome 扩展，并且不会重复自动启动标准或启动通知协议中所说的任何内容。

目前我在 gnome-session 中找到了使用这些扩展的代码。以下是我通过检查源代码能够确定的内容。在这里可以找到不同扩展的定义。在写完大部分内容之后，我发现这个链接解释了一些关于自动启动和 gnome-session 基本原理的东西。

Gnome 会话管理器将信息记录到 systemd 日志中。它记录以下信息（并非详尽无遗）：

当它启动一个自动启动应用程序时
当它无法启动自动启动应用程序时
当它无法停止应用程序以重新启动它时
当它跳过被条件禁用的应用程序时
启动成功后，GSM 会记录：“进入运行阶段”
当它从目录添加自动启动应用程序时
请注意，除了 .desktop 之外，还有.session和.directory文件。

## X-GNOME-Autostart-enabled

这似乎意味着它已被弃用。在任何情况下，自动启动标准中描述的 Hidden 属性看起来都提供了相同的功能。

## X-GNOME-Autostart-Phase

gnome-shell 识别以下阶段：

早期初始化
显示服务器
初始化
窗口管理器
控制板
桌面
其他任何东西，包括缺少的 X-GNOME-Autostart-Phase 属性都被认为处于“应用程序”阶段。
我找到了一个解释不同阶段的自述文件。启动分为7个阶段（GsmManagerPhase）：

GSM_MANAGER_PHASE_STARTUP 涵盖了 gnome-session 的内部启动，其中还包括启动 gconfd 和 dbus-daemon（如果它尚未运行）。Gnome-session 会显式启动它们，因为它需要它们用于自己的目的。
GSM_MANAGER_PHASE_EARLY_INITIALIZATION 是“正常”启动的第一阶段（即由 .desktop 文件而不是硬编码控制的启动）。它涵盖了通过 gnome-initial-setup 在 $HOME 中可能安装的文件，并且必须在 gnome-keyring 等其他组件使用这些文件之前完成。
GSM_MANAGER_PHASE_INITIALIZATION 涵盖了像 gnome-settings-daemon 和 at-spi-registryd 这样的低级东西，它们需要很早就运行（在显示任何窗口之前）。此阶段的应用程序可以使用 D-Bus 接口 (org.gnome.SessionManager.Setenv) 在 gnome-session 的环境中设置环境变量。这可以用于 $GTK_MODULES、$GNOME_KEYRING_SOCKET 等
GSM_MANAGER_PHASE_WINDOW_MANAGER 包括窗口管理器和合成管理器，以及在映射任何窗口之前必须运行的任何其他内容
GSM_MANAGER_PHASE_PANEL 包括任何永久占用屏幕空间的东西（通过 EWMH 支柱）。这是事物实际出现在屏幕上的第一阶段。
GSM_MANAGER_PHASE_DESKTOP 包括直接在桌面上绘制的任何内容（例如，nautilus）。
GSM_MANAGER_PHASE_APPLICATION 是其他一切（普通应用程序、托盘图标等）
Gnome 会话管理器知道其他几个阶段，可以在此处看到，但我认为您不能在自动启动启动器中使用它们：

GSM_MANAGER_PHASE_RUNNING
GSM_MANAGER_PHASE_QUERY_END_SESSION
GSM_MANAGER_PHASE_END_SESSION
GSM_MANAGER_PHASE_EXIT

## X-GNOME-Provides

允许应用程序定义一个互斥角色。例如。如果一个应用程序管理桌面，则不应以相同的角色启动第二个应用程序。常见的值是：

控制板
窗口管理器
文件管理器
我找到了有关所需组件的更深入的文档。

## X-GNOME-Autostart-startup-id

## X-GNOME-Autostart-Notify

也许与标准化的 [StartupNotify](http://standards.freedesktop.org/startup-notification-spec/startup-notification-0.1.txt) 有关？

## X-GNOME-AutoRestart

我的印象是，这意味着如果应用程序死了，它将自动重新启动，就像桌面管理器在崩溃时重新启动一样。

## X-GNOME-DBus-Name

## X-GNOME-DBus-Path

## X-GNOME-DBus-Start-Arguments


## X-GNOME-Autostart-discard-exec

从我在这里收集的信息来看，这似乎是 gnome-session 在内部使用的，以便在应用程序重新启动登录后清除已保存的会话数据。

## X-GNOME-Autostart-Delay（这是我在linux Mint 的 Cinnamon-session 中找到的）

## X-GNOME-DocPath（在gnome 管理员指南中找到）

指定当您从菜单项弹出菜单中选择关于应用程序名称的帮助时要显示的帮助文件。

## X-GNOME-Bugzilla-Bugzilla

## X-GNOME-Bugzilla-Product

## X-GNOME-Bugzilla-Component

## X-GNOME-Bugzilla-Version

## X-GNOME-WMName

## X-GNOME-WMSettingsModule


## X-GNOME-WMName

## X-GNOME-WMSettingsModule

## 其他非 gnome 特定的键

## Autostart-condition
xdg 邮件列表中有关此密钥的建议。

## StartupWMClass
大概是指可以用xprop检查的属性。此属性允许保持正在运行的应用程序与其桌面文件之间的连接。您可以通过运行xprop并单击应用程序的主窗口来检查应用程序的窗口类。例如，如果程序在其可执行文件中有一个图标，alt-tab 可能会显示该图标，而不是桌面文件中的图标。它也可能像所有 Java 应用程序一样将不同的程序错误地组合在一起。运行谷歌搜索“gnome3 dash StartupWMClass”会得到相当多的结果。在某些应用程序上，您可以添加--class命令行参数（例如 firefox 和据称是 gtk 应用程序）以分隔 alt-tab 图标。
