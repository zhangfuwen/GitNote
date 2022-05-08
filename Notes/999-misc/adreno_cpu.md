# adreno ftrace

https://www.cnblogs.com/lingjiajun/p/11913376.html


## Step 1 set buffer size

Connect USB and execute below commands one by one , some times all commands in batch may not work due to adb file system issues
Verify Buffer size is reflected as the one that is set


```bash
adb shell "echo 0 > /sys/kernel/debug/tracing/tracing_on"
adb shell "cat /sys/kernel/debug/tracing/tracing_on"
adb shell "echo 150000 > /sys/kernel/debug/tracing/buffer_size_kb"
adb shell "cat /sys/kernel/debug/tracing/buffer_size_kb"

adb shell "echo  > /sys/kernel/debug/tracing/set_event"
adb shell cat /sys/kernel/debug/tracing/set_event
adb shell "echo  > /sys/kernel/debug/tracing/trace"
adb shell cat /sys/kernel/debug/tracing/trace
adb shell sync
```




## Step 2 Enable below trace events


<summary>Enable below trace events</summary>

```bash
adb shell "echo 1 > /sys/kernel/debug/tracing/events/power/cpu_idle/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/power/cpu_frequency/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/power/cpu_frequency_switch_start/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/msm_low_power/*/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/sched/sched_switch/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/sched/sched_wakeup/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/sched/sched_wakeup_new/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/sched/sched_enq_deq_task/enable"

adb shell "echo 1 > /sys/kernel/debug/tracing/events/msm_bus/bus_update_request/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/clk/clk_set_rate/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/clk/clk_enable/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/clk/clk_disable/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/power/bw_hwmon_update/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/power/bw_hwmon_meas/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/power/memlat_dev_meas/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/power/memlat_dev_update/enable"

adb shell "echo 1 > /sys/kernel/debug/tracing/events/power/cpu_idle/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/power/cpu_frequency/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/power/cpu_frequency_switch_start/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/msm_low_power/*/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/sched/sched_switch/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/sched/sched_wakeup/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/sched/sched_wakeup_new/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/sched/sched_enq_deq_task/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/msm_low_power/cluster_enter/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/msm_low_power/cluster_exit/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/msm_low_power/cluster_pred_hist/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/msm_low_power/cluster_pred_select/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/msm_low_power/cpu_idle_enter/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/msm_low_power/cpu_idle_exit/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/msm_low_power/cpu_power_select/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/msm_low_power/cpu_pred_hist/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/msm_low_power/cpu_pred_select/enable"

adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/adreno_cmdbatch_queued/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/adreno_cmdbatch_submitted/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/adreno_cmdbatch_retired/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/kgsl_gpubusy/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/kgsl_pwr_request_state/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/kgsl_pwr_set_state/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/kgsl_pwrstats/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/kgsl_buslevel/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/kgsl_pwrlevel/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/kgsl_clk/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/kgsl_bus/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/kgsl_rail/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/dispatch_queue_context/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/kgsl_user_pwrlevel_constraint/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/kgsl_clock_throttling/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/kgsl_constraint/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/adreno_gpu_fault/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/adreno_cmdbatch_fault/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/adreno_cmdbatch_sync/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/kgsl_issueibcmds/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/kgsl_context_create/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/kgsl_context_destroy/enable"
adb shell "echo 1 > /sys/kernel/debug/tracing/events/kgsl/kgsl_context_detach/enable"
adb shell "echo 1 > /d/tracing/events/msm_bus/bus_update_request/enable"
adb shell "echo 1 > /d/tracing/events/msm_bus/bus_agg_bw/enable"
```



## Step 3 Verify the trace events and also remove any previous trace file

```bash
adb shell cat /sys/kernel/debug/tracing/set_event
adb shell rm /data/local/trace.txt
```


## Step 4:  Start Usecase, Execute below commands with usb and disconnect usb within 5 seconds

```bash
adb shell
cd /d/tracing
sleep 10 && echo 0 > tracing_on && echo "" > trace && echo 1 > tracing_on && sleep 10 && echo 0 > tracing_on && cat trace > /data/local/trace.txt &
```

## Step 5 : After 20 seconds, re-connect usb and pull the trace file

```bash
adb pull /data/local/trace.txt

```

# 其它sysfs文件的使用

## 设置gpu 频率

```bash

/sys/kernel/gpu # echo 587 > gpu_min_clock 

/sys/kernel/gpu # ls
gpu_available_governor  gpu_clock       gpu_governor   gpu_min_clock  gpu_tmu
gpu_busy                gpu_freq_table  gpu_max_clock  gpu_model
```

## 查看每个进程的显存占用


```bash

cat /sys/kernel/debug/kgsl/proc/1392/mem
```


# ahardware buffer慢的问题分析

## 1. 是普遍的吗？

是的，红米上也是。但是红米上pbuffer surface不慢。adreno上pbuffer surface也比texture慢。


**红米**

红米上第一和第三个数据没有高过4.40秒过。
第二个数据一直是高于4.40秒的。

```bash
on offscreen(pbuffer surface直接渲染)
1 round: time: 431108 us
2 round: time: 420680 us
3 round: time: 426077 us
4 round: time: 426393 us
5 round: time: 434032 us
6 round: time: 426161 us
7 round: time: 441449 us
8 round: time: 432911 us
9 round: time: 425257 us
10 round: time: 424511 us
total us: 4288579
on ahardwarebuffer
1 round: time: 448521 us
2 round: time: 461568 us
3 round: time: 433426 us
4 round: time: 440223 us
5 round: time: 422283 us
6 round: time: 446539 us
7 round: time: 442427 us
8 round: time: 459054 us
9 round: time: 454914 us
10 round: time: 448713 us
total us: 4457668
on texture
1 round: time: 442801 us
2 round: time: 408490 us
3 round: time: 405371 us
4 round: time: 434297 us
5 round: time: 435021 us
6 round: time: 437892 us
7 round: time: 436052 us
8 round: time: 429091 us
9 round: time: 453283 us
10 round: time: 428076 us
total us: 4310374

```

**adreno 650(587Hz锁频)**

```bash
on offscreen
1 round: time: 123494 us
2 round: time: 114145 us
3 round: time: 118861 us
4 round: time: 119680 us
5 round: time: 111774 us
6 round: time: 118367 us
7 round: time: 112398 us
8 round: time: 116392 us
9 round: time: 120694 us
10 round: time: 117300 us
total us: 1173105
on ahardwarebuffer
1 round: time: 108460 us
2 round: time: 114110 us
3 round: time: 115324 us
4 round: time: 114463 us
5 round: time: 111536 us
6 round: time: 116713 us
7 round: time: 116951 us
8 round: time: 116357 us
9 round: time: 110921 us
10 round: time: 113083 us
total us: 1137918
on texture
1 round: time: 80482 us
2 round: time: 89090 us
3 round: time: 83895 us
4 round: time: 84349 us
5 round: time: 71169 us
6 round: time: 95783 us
7 round: time: 74606 us
8 round: time: 88712 us
9 round: time: 76517 us
10 round: time: 81080 us
total us: 825683

```



## 2. 跟size有关吗？

下一项测试用的size是：

```c
#define VIEW_PORT_WIDTH 3750
#define VIEW_PORT_HEIGHT 1750
```

修改为3750x3750， 在adreno 650上测试5000帧：

pbuffer surface: 11.7s
ahardwarebuffer : 10.5s
texture: 6.9s
相差至少30%。

修改为3744x3744和3712x3712，得到的数据基本一致，所以可以说与alighment/stride无关。


<summary> log </summary>

```bash
on offscreen
1 round: time: 239200 us
2 round: time: 235986 us
3 round: time: 230568 us
4 round: time: 230193 us
5 round: time: 237294 us
6 round: time: 235746 us
7 round: time: 241780 us
8 round: time: 237943 us
9 round: time: 237922 us
10 round: time: 236801 us
11 round: time: 240831 us
12 round: time: 227555 us
13 round: time: 240831 us
14 round: time: 238398 us
15 round: time: 232454 us
16 round: time: 234113 us
17 round: time: 231645 us
18 round: time: 231582 us
19 round: time: 231839 us
20 round: time: 234578 us
21 round: time: 230020 us
22 round: time: 228352 us
23 round: time: 232705 us
24 round: time: 232300 us
25 round: time: 236544 us
26 round: time: 227925 us
27 round: time: 231249 us
28 round: time: 228276 us
29 round: time: 227464 us
30 round: time: 243549 us
31 round: time: 240960 us
32 round: time: 237399 us
33 round: time: 237119 us
34 round: time: 236078 us
35 round: time: 236671 us
36 round: time: 231312 us
37 round: time: 234462 us
38 round: time: 227629 us
39 round: time: 235564 us
40 round: time: 237280 us
41 round: time: 235940 us
42 round: time: 233894 us
43 round: time: 231069 us
44 round: time: 238023 us
45 round: time: 231004 us
46 round: time: 233870 us
47 round: time: 232869 us
48 round: time: 236491 us
49 round: time: 232334 us
50 round: time: 233229 us
total us: 11718840
on ahardwarebuffer
1 round: time: 214710 us
2 round: time: 208558 us
3 round: time: 211124 us
4 round: time: 206776 us
5 round: time: 207510 us
6 round: time: 206748 us
7 round: time: 217875 us
8 round: time: 210542 us
9 round: time: 214641 us
10 round: time: 212810 us
11 round: time: 209464 us
12 round: time: 210966 us
13 round: time: 212670 us
14 round: time: 203889 us
15 round: time: 207923 us
16 round: time: 208850 us
17 round: time: 210717 us
18 round: time: 214410 us
19 round: time: 209768 us
20 round: time: 214343 us
21 round: time: 209262 us
22 round: time: 209366 us
23 round: time: 210997 us
24 round: time: 208844 us
25 round: time: 203671 us
26 round: time: 204207 us
27 round: time: 203897 us
28 round: time: 214952 us
29 round: time: 213377 us
30 round: time: 217619 us
31 round: time: 216155 us
32 round: time: 215749 us
33 round: time: 207450 us
34 round: time: 209005 us
35 round: time: 212920 us
36 round: time: 213779 us
37 round: time: 211090 us
38 round: time: 217561 us
39 round: time: 217503 us
40 round: time: 216235 us
41 round: time: 213700 us
42 round: time: 212764 us
43 round: time: 209403 us
44 round: time: 211274 us
45 round: time: 212888 us
46 round: time: 209560 us
47 round: time: 211014 us
48 round: time: 214686 us
49 round: time: 205858 us
50 round: time: 212813 us
total us: 10561893
on texture
1 round: time: 156541 us
2 round: time: 136585 us
3 round: time: 137027 us
4 round: time: 138413 us
5 round: time: 131956 us
6 round: time: 136999 us
7 round: time: 140968 us
8 round: time: 141726 us
9 round: time: 141868 us
10 round: time: 134260 us
11 round: time: 137140 us
12 round: time: 139581 us
13 round: time: 140527 us
14 round: time: 143562 us
15 round: time: 139396 us
16 round: time: 140305 us
17 round: time: 140609 us
18 round: time: 141862 us
19 round: time: 140919 us
20 round: time: 136966 us
21 round: time: 141123 us
22 round: time: 140445 us
23 round: time: 135736 us
24 round: time: 144882 us
25 round: time: 144431 us
26 round: time: 145606 us
27 round: time: 134654 us
28 round: time: 135408 us
29 round: time: 139703 us
30 round: time: 140094 us
31 round: time: 137463 us
32 round: time: 135763 us
33 round: time: 139017 us
34 round: time: 142513 us
35 round: time: 131681 us
36 round: time: 136342 us
37 round: time: 136359 us
38 round: time: 134447 us
39 round: time: 133959 us
40 round: time: 134748 us
41 round: time: 136241 us
42 round: time: 134088 us
43 round: time: 131004 us
44 round: time: 138493 us
45 round: time: 141001 us
46 round: time: 141068 us
47 round: time: 140764 us
48 round: time: 137097 us
49 round: time: 144279 us
50 round: time: 140937 us
total us: 6946556
```


2. 有办法通过代码解决吗？

