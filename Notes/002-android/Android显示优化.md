# Android显示优化

Queue Stuffing: 队列塞满，指生产者生产太快，导致队列没有足够空间。此时会产生反压(back-pressure)，让生产者慢点生产。
frame pacing: 帧节奏控制，帧步调控制，指有故意放慢或加快提交帧的时间，来保证动画均匀。
stuttering: 口吃，结巴。这里指动画不均匀。