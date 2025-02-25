
linux内核通过调度器来决定将哪个任务分发给哪个CPU，通过cpufreq子系统来决定CPU的运行频率。cpufreq与调度器是不直接关联的。cpufreq会参考调度器的数据来决定升频还是降频。

CPU 频率的管理主要由 `cpufreq` 子系统负责。`cpufreq` 子系统提供了多种频率调节策略，如 `performance`（始终保持最高频率）、`powersave`（始终保持最低频率）、`ondemand`（根据系统负载动态调整频率）等。这些策略会根据系统的实时负载情况来调整 CPU 频率，而系统负载信息可以从 CFS 调度器的调度行为中间接获取。

# 用户态操作cpufreq

`cpufreq` （CPU 频率调节子系统）在一定程度上是可以从用户态进行操纵的。用户态程序可以通过多种方式来影响和控制 CPU 频率的设置，以下是一些常见的方法和途径：

### 1. 直接访问 sysfs 文件系统
在 Linux 系统中，`/sys/devices/system/cpu` 目录下包含了与 CPU 相关的各种信息和设置，`cpufreq` 相关的设置也在其中。具体来说：
- **查看和设置 CPU 频率调节策略**：`/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` 文件存储了每个 CPU 核心当前使用的频率调节策略。用户可以通过读取该文件内容来查看当前策略（如 `performance`、`powersave`、`ondemand` 等），也可以通过写入不同的策略名称来更改策略。例如，以 root 用户身份在终端中执行以下命令将 CPU 0 的频率调节策略设置为 `performance`：
```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```
- **设置 CPU 频率**：`/sys/devices/system/cpu/cpu*/cpufreq/scaling_setspeed` 文件允许用户设置 CPU 运行频率（前提是该频率在 CPU 支持的频率范围内）。不过，这需要 root 权限，并且不是所有的 CPU 和驱动都支持直接设置频率。例如，若要将 CPU 0 的频率设置为 2000000 kHz（2 GHz），可以执行以下命令（假设该频率在支持范围内）：
```bash
echo 2000000 | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
```

### 2. 使用命令行工具
- **`cpufrequtils`**：这是一个常用的工具包，包含了 `cpufreq-set` 等命令。`cpufreq-set` 可以用来查看和设置 CPU 频率调节策略以及频率。例如，使用以下命令将所有 CPU 核心的频率调节策略设置为 `ondemand`：
```bash
sudo cpufreq-set -g ondemand
```
使用以下命令将 CPU 0 的频率设置为 1500 MHz（假设该频率在支持范围内）：
```bash
sudo cpufreq-set -c 0 -f 1500000
```
- **`powerprofiles-daemon`**：在一些 Linux 发行版中，`powerprofiles-daemon` 提供了一种更高级的电源管理方式，用户可以通过设置不同的电源配置文件（如高性能、平衡、节能等）来间接影响 CPU 频率。例如，在 Ubuntu 系统中，可以使用 `powerprofilesctl` 命令来切换电源配置文件，从而改变 CPU 频率调节策略。

### 3. 编程实现
用户可以使用 C、Python 等编程语言编写程序，通过系统调用和文件 I/O 操作来访问上述 `sysfs` 文件系统中的相关文件，从而实现对 `cpufreq` 的控制。以下是一个简单的 Python 示例代码，用于查看 CPU 0 的当前频率调节策略：
```python
with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'r') as f:
    governor = f.read().strip()
    print(f"当前 CPU 0 的频率调节策略: {governor}")
```

需要注意的是，对 CPU 频率的操作通常需要 root 权限，并且不正确的操作可能会导致系统不稳定或硬件损坏。因此，在进行相关操作时，务必谨慎，并确保对 CPU 硬件和系统设置有充分的了解。 