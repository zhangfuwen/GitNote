## manifest库示例
一个最简单的manifest库可以这样去生成:


```
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


```
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


```

mkdir ~/myboard.git
git init --bare
cd ~/myrepo/adk1/board
git push ~/myboard.git master
```
这里我们在本地新建了一个git库，并把adk1/board的master分支推送了上去。
然后我们要修改manifest，让以后repo sync adk1/board的时候使用我们自己建的库，而不是从googlesource.com上下载，为示区别，我们使用自己项目的develop分支。


```
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
```
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


```
cd ~/myrepo/adk1/board
git branch -v
```

default字段
每个project都指定remote和revision有点麻烦，可以通过default指定一个remote和revision做为缺省设定。
比如在android开源项目中，多少project的remote都是aosp，而一个android发行版本的个个project都打了相同的tag，比如说是android_7.0.0-r1。
所以manifest.xml文件可以这样写：


```
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

