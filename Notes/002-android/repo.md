# repo命令介绍

Android开源项目是一个大项目。项目源代码有8GB大小。这样大的代码量用svn或是git管理都会感觉很吃力，可以预见checkout一个版本需要好长时间，好多人不停地提交代码，也会使代码的log变得难以阅读。

为了解决这个问题，Android项目将源代码分解成多个不同的子项目，并分别放在一个git库里来管理。这样每个子项目可以只关注本子项目的开发，而无需关注其他子项目的代码。

如何把这些子项目都下载到一个目录中，然后构建成一个Android安装镜像呢？Android用的是一个叫repo的命令。

repo命令是一些python脚本。它依赖一个manifest git库，这个manifest git库里只有一个文件，叫default.xml。这个xml文件里实际是一个子项目列表，记录了这个Android工程所依赖的子项目信息，包括每个子项目的git url、 下载到本地的目录、需要使用子项目的那个revision。

![repo architecture](/assets/06dd1f85b8c87ee4aeb92bcb4f6feb84c2143b3d.jpg)

# 命令基本用法

## 下载Android源代码

开始Android开发之前，需求先下载一套Android源代码。

## 安装repo工具

To install Repo:

Make sure you have a bin/ directory in your home directory and that it is included in your path:

```bash
$ mkdir ~/bin
$ PATH=~/bin:$PATH
```

Download the Repo tool and ensure that it is executable:


```bash
$ curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo
$ chmod a+x ~/bin/repo
```

## 下载manifest库

```bash
$ repo init -u https://android.googlesource.com/platform/manifest
```

## 从manifest库中取一个android版本


```bash
repo init -u https://android.googlesource.com/platform/manifest -b android-4.0.1_r1
```

或者


```bash
cd .repo/manifests
git checkout android-4.0.1_r1
```
这里，你可以读一下.repo/manifests文件夹下的default.xml代码。


##下载这个版本的android代码



```bash
repo sync
```


我们通过repo init初始化一个工程，通过repo sync把各个project都下载下来。其实其余的工作都可以通过git完成了。比如说project的commit、push、pull。

然而还是有一些场景使用repo命令会比较方便。

# 在所有的project中启动一个新分支
``repo start newbranchname``
# 对所有project执行命令


```bash
repo forall -c git checkout master
repo forall -c git git reset --hard HEAD
```

# manifest库示例

一个最简单的manifest库可以这样去生成:

```bash
# initialize a new bare git repo
mkdir ~/my_manifest.git
cd my_manifest.git
git init --bare

# initialize a new working copy
mkdir ~/my_manifest
cd ~/my_manifest
vim default.xml #写入内容
git add .
git commit -m "initial commit"
git push ~/my_manifest.git master

```

在default.xml中写入如下内容：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<manifest>
  <remote  name="aosp"
           fetch="http://android.googlesource.com/" />
  <project path="adk1/board" name="device/google/accessory/arduino"  remote="aosp" revision="master" />
  <project path="adk1/app" name="device/google/accessory/demokit" remote="aosp" revision="master" />

</manifest>
```

这个manifest库包含两个字项目，两个字项目有共同的`remote前缀`，即remote name="aosp"。两个子项目的实际remote url是拿这个前缀和自己的name字段拼接出来的，即分别为

```
http://android.googlesource.com/device/google/accessory/arduino.git
http://android.googlesource.com/device/google/accessory/demokit.git
```

这两个项目下载下来后存放的路径是adk1/board和adk1/app。下载下来后再git checkout到master这个commit。

## 下载这个库

建一个文件夹，在里面执行repo init和repo sync就可以把这两个字项目下载下来了：
```
mkdir ~/myrepo
repo init -u ~/my_manifest.git
repo sync
```

## 替换子项目
你现在可能要修改adk1/board中的代码，那你要自己建一个git库来管理这个子项目。那么我们可以这样做:

```bash
mkdir ~/myboard.git
git init --bare
cd ~/myrepo/adk1/board
git push ~/myboard.git master
```

这里我们在本地新建了一个git库，并把adk1/board的master分支推送了上去。
然后我们要修改manifest，让以后repo sync adk1/board的时候使用我们自己建的库，而不是从googlesource.com上下载，为示区别，我们使用自己项目的develop分支。

```bash
# 建立develop分支
cd ~/myrepo/adk1/board
git checkout -b develop
# 做一些修改
git add .
git commit -m "add develop branch"
git push ~/myboard.git develop

vi ~/myrepo/.repo/manifests/default.xml #修改的内容在后面
repo sync
```

修改后的default.xml如下：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<manifest>
  <remote  name="aosp"
           fetch="http://android.googlesource.com/" />
  <remote name="myhome" fetch="~/" />
  <project path="adk1/board" name="myboard"  remote="myhome" revision="develop" />
  <project path="adk1/app" name="device/google/accessory/demokit" remote="aosp" revision="master" />

</manifest>
```

这里我们主要增加了一个remote前缀为“~/"，即我们的home文件夹，名字设为myhome供后面project里引用。修改adk1/board这个project的remote为myhome, name为myboard。这样，我们得到的新子项目git url为~/myboard.git。revision字段设为develop，因而repo sync之后，我们得到的adk1/board是在develop分支上的。
可以通过以下命令查看：

```bash
cd ~/myrepo/adk1/board
git branch -v
```

default字段:

每个project都指定remote和revision有点麻烦，可以通过default指定一个remote和revision做为缺省设定。
比如在android开源项目中，多少project的remote都是aosp，而一个android发行版本的个个project都打了相同的tag，比如说是android_7.0.0-r1。
所以manifest.xml文件可以这样写：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<manifest>
  <remote name="myhome" fetch="~/" />
  <remote  name="aosp"
           fetch=".."
           review="https://android-review.googlesource.com/" />
  <default revision="refs/tags/android_7.0.0-r1"
           remote="aosp"
           sync-j="4" />
  <project path="adk1/board" name="device/google/accessory/arduino" />
  <project path="adk1/app" name="device/google/accessory/demokit" />
  <project path="adk2012/app" name="device/google/accessory/adk2012" />
  <project path="adk2012/board" name="device/google/accessory/adk2012_demo" />
  <project path="external/ide" name="platform/external/arduino-ide" />
  <project path="external/toolchain" name="platform/external/codesourcery" />

  <project path="adk1/board" name="myboard"  remote="myhome" revision="develop" />

</manifest>
```

这里还有一个`sync-j=“4”`字段，它的含义是执行repo sync时，相当于执行repo sync -j 4。-j 4表示下载时使用4个线程。这与参数make构建命令也支持，想必大家都用过。

由于下载都是一些网络下载，网络操作又会阻塞，所以可以尽量多用一些线程去下载，可以多于cpu物理线程数。
