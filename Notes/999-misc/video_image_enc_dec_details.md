
色彩空间:  RGB YUV

音频采样率，采样位数

声道：单声道，立体声，四声道立体声

码率：每秒数据量，bps为单位，= 采样率 X 采样位数 X 声道数

视频压缩编码：ITU国际电传视讯联盟， 主导H26x(1,2,3,4,5)。 ISO/MPEG(Motion Picture Experts Group)主导标准MPEG(1,2,3,4)。H264为两个组织联合制定。

图像组GOP(Group Of Pictures):指连续变化不大的帧，其第一帧为关键帧， 叫IDR帧（Instantaneous Decoding Refresh), 解码器遇IDR帧可丢弃前面所有缓存的参考帧。IDR帧必为I帧。

I帧：帧内编码帧。就是一个完整帧。

P帧：前向预测编码帧。是一个非完整帧，通过参考前面的I帧或P帧生成。

B帧：双向预测内插编码帧。参考前后图像帧编码生成。B帧依赖其前最近的一个（TODO:也许不是最近的?)I帧或P帧及其后最近的一个P帧。

DTS&PTS: Decoding TimeStamp, Presentation TimeStamp, 解码和显示的时间戳。B帧两者不一致。

H264采用YUV色彩空间

YUV存储方式分为两大类：planar 和 packed。

planar：先存储所有Y，紧接着存储所有U，最后是V；

packed：每个像素点的 Y、U、V 连续交叉存储。

最常用的是YUV420, 为planar存储方式。

YUV420P：三平面存储。数据组成为YYYYYYYYUUVV（如I420）或YYYYYYYYVVUU（如YV12）。

YUV420SP：两平面存储。分为两种类型YYYYYYYYUVUV（如NV12）或YYYYYYYYVUVU（如NV21）

音频编码：

PCM(Pulse Code Modulation), 脉冲编码调制。

AAC(Advanced Audio Coding), 高级音频编码。MP4中常用。分为用于存储的ADIF和用于传输的ADTS(Audio Data Interchange Format/Transport Stream)。具体格式：https://blog.csdn.net/wlsfling/article/details/5876016

音视频容器：mp4, rmvb, avi, mkv, mov。mp4支持H264,H265等和AAC, MP3等。