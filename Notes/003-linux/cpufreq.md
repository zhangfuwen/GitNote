
linux内核通过调度器来决定将哪个任务分发给哪个CPU，通过cpufreq子系统来决定CPU的运行频率。cpufreq与调度器是不直接关联的。cpufreq会参考调度器的数据来决定升频还是降频。

CPU 频率的管理主要由 `cpufreq` 子系统负责。`cpufreq` 子系统提供了多种频率调节策略，如 `performance`（始终保持最高频率）、`powersave`（始终保持最低频率）、`ondemand`（根据系统负载动态调整频率）等。这些策略会根据系统的实时负载情况来调整 CPU 频率，而系统负载信息可以从 CFS 调度器的调度行为中间接获取。

