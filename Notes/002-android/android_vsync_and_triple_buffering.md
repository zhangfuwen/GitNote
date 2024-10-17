# Android的vsync和三缓冲

## 传统双缓冲

在最传统的显示系统中，我们通过双缓冲来解决画面撕裂问题：

![](/assets/res/2021-12-24-11-46-29.png)

生产者在后缓冲中写入像素，显示控制器读取前缓冲区中的内容来显示。
生产者生产完一帧，显示器消费完一帧后，两者交换。

这样不会产生画面撕裂问题，一切看起来都　很完美。


但是如果生产者在任意时间生产，并且不保证16.6ms内能完成生产，每一帧的生产时间不固定，那这个节奏就会被打破，产生丢帧。

图中生产活动包括CPU和GPU两个阶段。

![](/assets/res/2021-12-24-11-57-31.png)

## Vsync

Android 4.1首先解决生产者时间同步问题，即让上一帧显示完后，生产者就开始生产。所以操作系统的底层会把vsync信号向上传递给应用，应用收到信号后，立即开始进行绘制。

![](/assets/res/2021-12-24-11-58-46.png)

但是这在某次生产活动没有如期完成的情况下会有很大问题：

![](/assets/res/2021-12-24-12-00-52.png)

即CPU和GPU是严格顺序执行的，CPU要等GPU做完再能生产下一帧。这种串行是不好的。

## 三缓冲

要想并行，就需要在多一个缓冲区，让同一时刻，CPU使用缓冲区A, GPU使用缓冲C, 消费者使用缓冲B。如下面的第二条vsync线与第三条vsync线之间的情况所示：

![](/assets/res/2021-12-24-14-12-42.png)

生产者无法及时生产，消费者就无法及时消费，第一次vsync时的jank不可避免。
但CPU与GPU两道生产工序充分并行，可以减轻一次渲染不及时产生和累积影响。

三缓冲并非常态机制，只有出现jank后，才需求第三个，所以也不会大量占用显存。

## vsync是主动申请的

当应用没有更新界面时，给他发送vsync没啥必要。白白唤醒一次进程。

在Choreographer中的FrameHandler.handleMessage中，会执行doScheduleVsync()，让系统后续向该应用发送vsync。应用在收到vsync时才开始执行。

## Android UI的绘制流程

Android有一个main thread,　也叫ui thread。
开发时有一个函数叫`runOnUiThread`，它的作用是把一个Runable投递到main thread的looper来执行：
[代码](https://android.googlesource.com/platform/frameworks/base/+/master/core/java/android/app/Activity.java)

```java
   public final void runOnUiThread(Runnable action) {
        if (Thread.currentThread() != mUiThread) {
            mHandler.post(action);
        } else {
            action.run();
        }
    }
```

有一些事情只能在UI Thread上做，那就更新`视图树`上的任何东西，如修改button上的文字、button的大小等。

修改button上的文字并不立即产生视觉效果。当我们执行`button.setText("xxx")`时，我们只是修改了`视图树`这个数据结构。`视图树`上的变化反应到屏上，需要经过Measure, Layout, Draw三个步骤。

Choreographer的onVsync回调会触发UI Thread的绘制动作，过程如下图：　

![](/assets/res/2021-12-24-15-07-38.png)

图片来源：https://androidperformance.com/2019/10/22/Android-Choreographer/

查看Choreographer的代码可以发现，`onVsync`触发了doAnimation和doDraw两种主要方法，这两个方法都是执行各自的callback列表中的callback。
callback列表的条目是通过`postDrawCallback`和`postAnimationCallback`两个方法填加的。对于draw而言，　`postDrawCallback`是由ViewRootImpl调用的，传入一个Runnable，里面包的是ViewRootImpl的`doFrame`方法。也就是说，Choreographer在收到Vsync时，会调用ViewRootImpl.doFrame()。而`doFrame`的工作就是处理输入、measure、layout、draw的过程。measure、layout、draw都是在间接调用的performTraverals这个方法里完成的，performTraversals是一个非常长的函数。[代码](https://android.googlesource.com/platform/frameworks/base/+/a175a5b/core/java/android/view/ViewRootImpl.java)

对于draw这个动作，ViewRootImpl有一个draw方法，它分为硬件绘制和软件绘制。硬件绘制会把mView交给mHardwareRenderer。软件绘制则是lockCanvas、绘制、unlockAndPost的过程。这个过程大概率不涉及第三个缓冲区。

在使用GPU进程渲染时，