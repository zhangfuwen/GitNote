
# Fun中文输入法

我注意到一般的linux发行版默认装的输入法框架都是ibus。ibus上也有一些中文输入法，但是可能谈不上好用。从我自己的角度看，它用五笔输入法，也有拼音输入法，但是美中不足的是没有一个五笔拼音混输的输入法。好多五笔输入法的用户并不能做到所有的汉字都用五笔输入（比如说我），五笔拼音混输就比较重要。

另外还有一个就是fcitx也好，ibus也好，都没有语音输入法。在一个比较安静的环境下，使用语音输入可能更有效率一些。先用语音说一遍，然后再微调一下，效率会提升不少。fcitx实际是有语音输入法的，但是现在好像只支持UOS，UOS咱不太愿意用。

基于以上原因，我开发了一个基于ibus的输入法，叫ibus-fun。这个输入法支持五笔、拼音、五笔拼音混输、语音输入、自定义快捷输入。

五笔输入采用的是86版和98版五笔输入法，可以选。
拼音输入法来自google的中文拼音输入法。
五笔拼音混输则是五笔和拼音候选词的合并。
语音输入采用的是阿里云的一句话识别API。这个API是收费的。我现在购买了一元3万次的新手包。其他同学想使用的话购买一下，然后把access id和access secret输入到配置界面里就可以使用了。
自定义快捷输入就是一个简单的bash命令执行。比如说你想在输入`rq`两个字母的时候返回当天的日期，就可以在value字段填入`date +%Y-%m-%d`。
目前整个输入法还不太稳定，有时会崩溃，还没找到原因。初步看可能跟拼音输入法有关，关闭拼音输入功能就可以保持稳定。欢迎有兴趣的同学试用，甚至帮助解决问题。有任何问题都可以随时交流。


项目网址：https://github.com/zhangfuwen/IBusFunIM

deb包下载地址：https://github.com/zhangfuwen/IBusFunIM/releases


# Intruction

一个Linux下的基于IBus的,支持五笔拼音混输, 支持单纯拼音输入,支持五笔输入,还支持语音输入的输入法.


Code under pinyin folder and dictionary file res/dict_pinyin.dat come from [Google pinyin IME](https://android.googlesource.com/platform/packages/inputmethods/PinyinIME).



# Build and Install

这是一个cmake工程,所有用正常的cmake编译就行了.

```bash
mkdir build
cd build
cmake ..
make
sudo make install
```

你也可以先构建一个deb包, 然后再安装deb包:

```bash
mkdir build
cd build
cmake ..
make
cpack
sudo dpkg -i audio_ime-1.1-Linux.deb

```

# Snapshots

## 五笔拼音混输界面

五笔和拼音可以单独打开和关闭, 关闭五笔就是一个拼音输入法, 关闭拼音就是一个五笔输入法.

五笔可以选择86版五笔还是98版五笔.

![输入界面](/assets/3_input.png)

## 语音输入界面

语音输入, 使用的是阿里云的一句话识别API, 需要用户自己花一块钱买一个一年3万次的服务包. 并在配置界面输入id和secret.
后续考虑把我自己的置入. 语音输入激活的快捷键是`ctrl + \``.

![语音输入界面](/assets/4_input_by_speech.png)

## 输入id和secret的界面

![setup界面](/assets/5_setup.png)

## 简单的配置界面

可以选择打开和关闭拼音或五笔, 可以选择候选词排列的方向.

![配置界面](/assets/1_config.png)

![五笔配置界面](/assets/2_config_wubi.png)

## 快速输入配置界面

![快速输入配置](/assets/6_fast_input.png)





