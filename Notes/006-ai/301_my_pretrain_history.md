---
tags:
  - pretrain
  - llm
  - python
title: LLM Pretrain过程记录
---

# my pretrain history

# pretrain

```
.py:2134] 2024-10-02 22:46:14,378 >> ***** Running training *****
[INFO|trainer.py:2135] 2024-10-02 22:46:14,378 >>   Num examples = 16,527,707
[INFO|trainer.py:2136] 2024-10-02 22:46:14,378 >>   Num Epochs = 3
[INFO|trainer.py:2137] 2024-10-02 22:46:14,378 >>   Instantaneous batch size per device = 8
[INFO|trainer.py:2140] 2024-10-02 22:46:14,378 >>   Total train batch size (w. parallel, distributed & accumulation) = 8
[INFO|trainer.py:2141] 2024-10-02 22:46:14,378 >>   Gradient Accumulation steps = 1
[INFO|trainer.py:2142] 2024-10-02 22:46:14,378 >>   Total optimization steps = 6,197,892
[INFO|trainer.py:2143] 2024-10-02 22:46:14,380 >>   Number of trainable parameters = 1,235,814,400
```

```
per_device_train_batch_size: 1  
gradient_accumulation_steps: 8  
learning_rate: 1.0e-4  
num_train_epochs: 3.0  
lr_scheduler_type: cosine  
warmup_ratio: 0.1  
bf16: true
```

16,527,707 * 3 /8 = 6,197,892

25gb


py:2134] 2024-10-04 08:58:31,915 >> ***** Running training *****
[INFO|trainer.py:2135] 2024-10-04 08:58:31,915 >>   Num examples = 826,279
[INFO|trainer.py:2136] 2024-10-04 08:58:31,915 >>   Num Epochs = 3
[INFO|trainer.py:2137] 2024-10-04 08:58:31,915 >>   Instantaneous batch size per device = 1
[INFO|trainer.py:2140] 2024-10-04 08:58:31,915 >>   Total train batch size (w. parallel, distributed & accumulation) = 8
[INFO|trainer.py:2141] 2024-10-04 08:58:31,915 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:2142] 2024-10-04 08:58:31,915 >>   Total optimization steps = 309,852
[INFO|trainer.py:2143] 2024-10-04 08:58:31,916 >>   Number of trainable parameters = 1,235,814,400

## 输出格式

![](assets/Pasted%20image%2020241008123149.png)

## 训练资源占用


### 100M 模型/ V100

V100是15 TFLOPS FP32算力。
1. 当cutoff_len为1时，算力没有跑满，内存瓶颈。
2. 当cutoff_len为8192时，算力应该差不多跑满了。此时算力为6*（参数量）* token数 * 2.34(speed) = 11T。接近15T.(没有用到Tensor Core，seq_len太多，在attention引入了更多计算)
3. 当cutoff_len为1024，gradient_accumulation_steps=8时，理论上速度与上一条相同，但实际慢了一些，可能是因为多了7轮内存访问。
4. 当cutoff_len为1024, batch_size为8时，因为seq_length小，计算量小了些的缘故。按6 * 100 * 1024 * 8 * 3.3 = 16T，比15T大一些。

| task         | fp16     | speed        | num_gpu | batch size per gpu | gradient_accumulation_steps | cutoff_len | gpu memory usage(GB) | memory bandwidth |
| ------------ | -------- | ------------ | ------- | ------------------ | --------------------------- | ---------- | -------------------- | ---------------- |
| pretrain     | true     | 12.9 it/s    | 1       | 1                  | 1                           | 1          | 2.579                |                  |
| pretrain     | true     | 12.7 it/s    | 1       | 1                  | 1                           | 64         | 2.617                |                  |
| pretrain     | true     | 12.4 it/s    | 1       | 1                  | 1                           | 128        | 2.729                |                  |
| pretrain     | true     | 12.4 it/s    | 1       | 1                  | 1                           | 256        | 2.951                |                  |
| pretrain     | true     | 12.4 it/s    | 1       | 1                  | 1                           | 512        | 3.183                |                  |
| pretrain     | true     | 12.3it/s     | 1       | 1                  | 1                           | 1024       | 4.5                  |                  |
| pretrain     | true     | 9.03 it/s    | 1       | 1                  | 1                           | 2048       | 6.583                |                  |
| pretrain     | true     | 5 it/s       | 1       | 1                  | 1                           | 4096       | 9.647                |                  |
| pretrain     | true     | 2.34 it/s    | 1       | 1                  | 1                           | 8192       | 18.067               |                  |
|              |          |              |         |                    |                             |            |                      |                  |
|              |          |              |         |                    |                             |            |                      |                  |
| ==pretrain== | ==true== | ==12.3it/s== | ==1==   | ==1==              | ==1==                       | ==1024==   | ==4.5==              |                  |
| ==pretrain== | ==true== | ==1.82it/s== | ==1==   | ==1==              | ==8==                       | ==1024==   | ==4.529==            |                  |
| ==pretrain== | ==true== | ==1.2s/it==  | ==1==   | ==1==              | ==16==                      | ==1024==   | ==4.529==            |                  |
| pretrain     | true     | 2.26 it/s    | 1       | **12**             | 1                           | 1024       | 26.353               |                  |
| **pretrain** | **true** | **3.3it/s**  | **1**   | **8**              | **1**                       | **1024**   | **18.8**             | **70%**          |
| pretrain     | true     | 2.5s/it      | 1       | 8                  | **8**                       | 1024       | 18.0                 |                  |
| pretrain     | true     | 4.8s/it      | 1       | 8                  | **16**                      | 1024       | 18.0                 |                  |

A100:

1. Training with DataParallel so batch size has been adjusted to: 8, 所以没法把batch size设成16或32，通过修改gradient_accumulation_steps来提高并行度，没有走作用。
2. 计算算力是：6 * 100 * 1024 * 8 * 6.99 = 34 TFLOPS，A100的fp engine是19.5 FLOPS，是不够的，主要用的是tensor core。
3. 看起来是内存瓶颈了，50%的内存带宽是1TB/s，1s计算的token数是1024 * 8 * 6.99 = 57,000个， 平均一个token访存17.5GB。这和memory usage是对得上的。但这个内存占用是1024 * 8才用到的，并不应该每个token都用到所有其他的token相关的内存。反向传播的内存访存是非常大的？怎么算？
4. 对于较小的模型来讲，gradient_accumulation_steps应该设置为1 ，设大的主要作的是减少权重更新次数？但小模型权重更新写内存本来就不多。

| task         | fp16     | speed     | num_gpu | batch size per gpu | gradient_accumulation_steps | cutoff_len | gpu memory usage(GB) | memory bandwidth | tensor core | fp engine |
| ------------ | -------- | --------- | ------- | ------------------ | --------------------------- | ---------- | -------------------- | ---------------- | ----------- | --------- |
| **pretrain** | **true** | 6.99 it/s | **1**   | **8**              | **1**                       | **1024**   | **18.8**             | 50%              | 12.9%       | 3%        |
| **pretrain** | **true** | 1.21 s/it | **1**   | **8**              | **8**                       | **1024**   | **18.8**             | 50%              | 12.9%       | 3%        |

### 1B 模型/ V100 *6

| num_gpu | batch size per gpu | gradient_accumulation_steps | cutoff_len | gpu memory usage(GB) | 算力  |
| ------- | ------------------ | --------------------------- | ---------- | -------------------- | --- |
| 6       | 16                 | 1                           | 1024       |                      |     |
|         |                    |                             |            |                      |     |
|         |                    |                             |            |                      |     |

### 1B 模型/ A100 

| task     | speed(s/it) | num_gpu | batch size per gpu | gradient_accumulation_steps | cutoff_len | gpu memory usage(GB) | memory bandwidth | fp32 | fp32算力(TFLOPS) |
| -------- | ----------- | ------- | ------------------ | --------------------------- | ---------- | -------------------- | ---------------- | ---- | -------------- |
| pretrain | 8.9         | 1       | 16                 | 1                           | 1024       | 42.73                | 10%(200GB/s)     | 90%  | 17.6           |
|          |             |         |                    |                             |            |                      |                  |      |                |
|          |             |         |                    |                             |            |                      |                  |      |                |
forward算力：2.4GFLOPS * 16 * 1024 = 38.4 TFLOPS(约需2s)，如果gradient_accumulation_steps设成8,那就是19s左右。
