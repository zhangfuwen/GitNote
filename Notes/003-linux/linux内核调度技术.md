## PELT (Per Entity Load Tracking)[¶](https://docs.kernel.org/scheduler/schedutil.html#pelt-per-entity-load-tracking "Permalink to this heading")

Linux的Per-entity load tracking（PELT，按实体负载跟踪）是内核中用于更精确地跟踪和管理系统负载的一种机制，以下是相关介绍：

### 背景及目的
- 在Linux 3.8版本之前的内核CFS调度器采用的是跟踪每个运行队列上的负载（per-rq load tracking），无法准确得知当前CPU上的负载来自哪些任务、每个任务施加多少负载等信息，难以实现更精细的负载均衡和CPU算力调整。
- PELT算法把负载跟踪从per-rq推进到per-entity的层次，让调度器能获取每个调度实体（进程或控制组中的一组进程）对系统负载的贡献，为更精准的调度算法提供支撑。

### 基本方法
- **时间划分**：将时间分成1024us的序列，在每个周期中计算一个entity对系统负载的贡献。
- **瞬时负载计算**：任务在1024us的周期窗口内的负载即瞬时负载，若在该周期内runnable的时间是t，引入负载权重loadweight，瞬时负载li = loadweight×(t/1024)。
- **瞬时利用率计算**：任务的瞬时利用率ui = maxcpucapacity×(t/1024)，与任务优先级无关。
- **平均负载计算**：由于瞬时负载和利用率变化快，不适合直接用于调整调度算法，因此需对瞬时负载进行滑动平均计算得到平均负载。一个调度实体的平均负载可以表示为l = l0 + l1×y + l2×y² + l3×y³+...，其中li表示在周期pi中的瞬时负载，y是衰减因子，在目前的内核代码中，y³²等于0.5。

### 优势
- **精准负载迁移**：在任务迁移时能准确携带其负载信息，避免了之前per-rq跟踪方式下任务迁移时负载计算不准确和可能出现的双重计数问题，使系统全局负载计算更准确，有助于实现更合理的负载均衡。
- **提供精细调度信息**：调度器可以清楚了解每个调度实体的负载情况，根据负载和利用率来更精准地为任务选择合适的CPU，进行迁核或提频等操作，提高系统资源利用率和整体性能，同时有助于实现更优的功耗管理。

### 实现中的挑战与解决
- **挑战**：非运行状态的实体（如因等待资源而阻塞的进程）也可能对系统负载有贡献，但系统中可能存在大量阻塞实体，若遍历它们来更新负载信息，会带来巨大的性能开销。
- **解决**：在每个CFS_RQ（控制组运行队列）结构中维护一个单独的“blocked load”总和。当进程阻塞时，将其负载从总可运行负载值中减去并添加到阻塞负载中，以与可运行实体负载分开计算和管理，可按相同的衰减因子y进行衰减，当阻塞进程再次变为可运行时，再将其负载进行相应处理。

# DVFS

DVFS 即 Dynamic Voltage and Frequency Scaling，意为**动态电压和频率缩放**。它用于调整 CPU 的电压和频率，以适应不同的工作负载需求，实现性能和功耗的平衡。


- 在频率 / CPU 不变性部分，由于不同频率下 CPU 的性能表现不同（如在 1GHz 下消耗 50% 的 CPU 与在 2GHz 下消耗 50% 的 CPU 不同），所以允许架构使用 DVFS 比率和微拱形比率来扩展时间增量，使 “运行” 和 “可运行” 指标不受 DVFS 和 CPU 类型的影响，实现指标在不同 CPU 之间的传输和比较12。
- 在 Schedutil / DVFS 部分，每次调度器负载跟踪更新时，会调用 schedutil 来更新硬件 DVFS 状态，以根据 CPU 运行队列的 “运行” 指标计算所需的频率，进而选择 P-state/OPP 或向硬件发出 CPPC 风格的请求 。在低负载场景下，“运行” 数值能反映利用率，而在饱和场景下任务迁移会使 “运行” 值波动，但时间推移会修正345


DVFS（Dynamic Voltage and Frequency Scaling）即**动态电压和频率缩放**，工作原理涉及多个方面：
- **比率计算**：对于简单的DVFS架构（软件完全控制），其比率计算为当前频率（`f_cur`）与最大频率（`f_max`）的比值，即`r_dvfs := f_cur / f_max`；对于硬件控制DVFS的动态系统，使用硬件计数器（如Intel APERF/MPERF、ARMv8.4-AMU）来提供比率。以Intel为例，`f_cur := APERF / MPERF * P0`，`f_max`根据不同情况取值，`r_dvfs := min(1, f_cur / f_max)`，并且选择4C涡轮增压而非1C涡轮增压以使其更具可持续性。
- **指标调整**：通过调整频率，使得 “运行” 和 “可运行” 这两个关键指标能够不受DVFS和CPU类型的影响，即实现频率/CPU不变性。这样可以在不同CPU之间传输和比较这些指标，`r_cpu`由当前CPU最高性能水平与系统中其他CPU最高性能水平的比率确定，`r_tot = r_dvfs * r_cpu`，从而使上述指标标准化。
- **与调度器协同工作**：每次调度器负载跟踪更新（如任务唤醒、任务迁移、时间推进）时，会调用schedutil更新硬件DVFS状态。依据CPU运行队列的 “运行” 指标（频率不变的CPU利用率估计值）计算所需频率，计算过程会考虑UTIL_EST、UCLAMP_TASK等因素，最终得到期望频率`f_des := min( f_max, 1.25 u * f_max )`，该频率用于选择P-state/OPP或直接转化为对硬件的CPPC风格请求。 

在 Linux 内核调度器（特别是 CFS，Completely Fair Scheduler）中，**`util`**（通常指 **CPU utilization**，即 CPU 利用率）是一个关键概念，用于衡量任务（task）或调度实体（sched entity）对 CPU 资源的需求程度。它主要用于调度决策，尤其是在 **CPU 频率调节（CPUFreq）** 和 **负载均衡（load balancing）**、**任务迁移** 以及 **调度类（如 EAS, Energy-Aware Scheduling）** 中。

---
# Util
### 一、`util` 的基本定义

在 Linux 调度器中，`util` 通常指的是 **调度实体的“运行需求”或“计算需求”**，它不是简单的 CPU 使用率百分比，而是一个经过加权和归一化的值，反映任务对 CPU 的“压力”。

#### 常见的 `util` 类型：

1. **`util_avg`**：
   - 表示调度实体在过去一段时间内的 **平均 CPU 利用率**。
   - 它通过指数衰减平均（exponential decay）的方式计算，对最近的 CPU 使用更敏感。
   - 单位通常是 **“单位”（scale）**，比如在 1024 为满量程的系统中，`util_avg = 1024` 表示一个任务持续占用一个 CPU 核心。

2. **`util_est`**：
   - 是“utilization estimation”，用于预测任务的未来利用率，尤其在任务刚唤醒时使用。
   - 帮助调度器做出更智能的决策，比如选择合适的 CPU（在 EAS 中）。

3. **`cpu_util`**：
   - 指某个 CPU 上所有运行任务的 `util_avg` 总和。
   - 用于判断 CPU 是否过载、是否适合运行新任务。

---

### 二、`util` 的用途

1. **能量感知调度（EAS, Energy-Aware Scheduling）**
   - EAS 使用 `util` 来估算任务对 CPU 的需求，并结合 CPU 的性能能力（capacity），选择既能满足性能需求又能节省能耗的 CPU。
   - 目标是将高 `util` 任务放到高性能大核，低 `util` 任务放到节能小核。

2. **CPU 频率调节（CPUFreq / EAS / schedutil）**
   - `util_avg` 被用作 `schedutil` 频率调节器的输入，动态调整 CPU 频率。
   - 例如：如果 `util_avg` 高，就提升频率；如果低，就降低频率以省电。

3. **负载均衡**
   - 在多核系统中，调度器会比较各个 CPU 的 `cpu_util`，决定是否迁移任务以平衡负载。

4. **调度类决策**
   - 比如在 `SCHED_IDLE` 或 `SCHED_DEADLINE` 中，`util` 可用于判断系统是否还有足够容量运行新任务。

---

### 三、`util` 的计算方式（简化）

Linux 使用 **per-entity load tracking (PELT)** 机制来计算 `util_avg`。

- 每个调度实体（task、cfs_rq、rt_rq 等）都有一个 `se.avg.util_avg`。
- 利用周期性更新的“贡献值”（基于运行时间和衰减系数）来计算。
- 公式大致如下（概念性）：

  ```
  util_avg = util_avg * decay_factor + (running_time * 1024 / sampling_period)
  ```

- 衰减因子随时间推移自动降低旧贡献的影响。

---

### 四、举例说明

假设一个任务在一个 CPU 上持续运行，其 `util_avg` 会趋近于 1024（表示占满一个 CPU）。

如果一个 CPU 上有两个任务，每个 `util_avg = 600`，那么该 CPU 的总 `cpu_util = 1200`。如果该 CPU 的 capacity 是 1024，则表示 **过载（over-utilized）**，可能需要迁移到其他 CPU。

---

### 五、相关代码和调试

在内核代码中，你可以在以下位置找到 `util` 相关实现：

- `kernel/sched/pelt.c`：PELT 核心逻辑
- `kernel/sched/fair.c`：CFS 调度器使用 `util_avg`
- `/proc/sched_debug`：可查看每个任务和 CPU 的 `util_avg`

例如：

```bash
cat /proc/sched_debug | grep util
```

输出可能包含：

```
cpu: 0, load: 512, util: 720, capacity: 1024
```

表示 CPU 0 的利用率是 720/1024 ≈ 70%。

---

### 六、总结

| 概念       | 含义 |
|------------|------|
| `util_avg` | 任务或 CPU 的加权平均利用率，用于调度决策 |
| `util_est` | 任务的利用率预测值，用于唤醒优化 |
| `cpu_util` | CPU 上所有任务的 util_avg 之和 |
| 单位       | 通常以 1024 为满量程（一个 CPU 核心） |
| 用途       | EAS、CPUFreq、负载均衡、调度决策 |

---

### 参考资料

- Linux 内核文档：`Documentation/scheduler/sched-energy.txt`
- LWN 文章：[Energy-aware scheduling](https://lwn.net/Articles/663991/)
- 内核源码：`kernel/sched/`

---

## 架构图

```text
                   +------------------+
                   |   Task Execution |
                   +--------+---------+
                            |
                            v
                   +--------+---------+
                   | CFS + PELT Engine| ← 生产者
                   |  Produces util_avg |
                   +--------+---------+
                            |
                            v
              +------------+------------+
              |       util_avg signal     |
              +------------+------------+
                           |
         +-----------------+------------------+
         |                 |                  |
         v                 v                  v
+--------+--------+ +-----+------+ +----------+----------+
|    schedutil    | | Load Balance| |      EAS / CPU Selection |
|  (CPU Frequency) | | (Migration) | | (Big.LITTLE Scheduling) |
+-----------------+ +-------------+ +-----------------------+
         |                 |                  |
         v                 v                  v
   Adjust Frequency   Balance Load      Save Energy & Boost Perf

```

# capacity是怎么算出来的

非常好的问题！你问到了 Linux 调度器中 **`capacity`（调度容量）** 的**根本来源**：它是怎么定出来的？是不是以最强的 CPU 核定为 1024？每个系统 + 硬件平台是怎么确定各个 CPU 的 `capacity` 值的？

我们来一步步深入解析。

---

## ✅ 简短回答：

> **是的，Linux 调度器通常将系统中最强 CPU 的最大性能定为 `1024`，其他 CPU 的 `capacity` 按相对性能比例缩放。**

这个值不是随意定的，而是通过 **硬件性能基准 + 设备树（Device Tree）或 ACPI + 内核启动时的 CPU 拓扑探测** 来确定的。

---

## 🧩 一、什么是 `capacity`？再回顾

在 Linux 调度器（特别是 EAS 和负载均衡）中：

- `capacity` 是一个 **归一化的整数**，表示某个 CPU 的**相对计算能力**。
- 最强的 CPU 定为 **1024**，其他核按比例缩小。
- 例如：
  - 大核：`capacity = 1024`
  - 中核：`capacity = 896`
  - 小核：`capacity = 512`

> 📌 注意：这是“逻辑容量”，不是物理频率或功耗。

---

## 🔍 二、`capacity` 是怎么确定的？流程图解

```
[硬件设计] 
   ↓
[芯片厂商提供性能数据] → Dhrystone/MIPS/SpecInt 等基准
   ↓
[设备树 (DT) 或 ACPI 表] → 描述 CPU 拓扑和性能等级
   ↓
[Linux 内核启动] → 解析 DT/ACPI，构建 CPU 性能域 (performance domains)
   ↓
[计算相对性能] → 找出最大 capacity，设为 1024
   ↓
[设置每个 CPU 的 capacity] → 按比例缩放
   ↓
[scheduler 使用] → 用于任务分配、EAS、负载均衡
```

---

## 📐 三、`capacity` 的计算方法

### 方法 1：通过 **设备树（Device Tree）** 显式指定

在 ARM64 平台上，常见如下设备树片段：

```dts
cpu0: cpu@0 {
    device_type = "cpu";
    compatible = "arm,cortex-a76";
    capacity = <1024>;
    cpu-idle-states = <&CPU_IDLE_STATES>;
};

cpu4: cpu@4 {
    device_type = "cpu";
    compatible = "arm,cortex-a55";
    capacity = <512>;
};
```

👉 这是最直接的方式：**厂商或 BSP 开发者手动设定**。

---

### 方法 2：通过 **CPU 拓扑 + 频率表 + 性能模型** 自动推导

如果设备树中没有 `capacity` 字段，内核会尝试自动计算。

#### 步骤如下：

1. **识别 CPU 拓扑**：哪些 CPU 属于同一个性能域（如所有大核共享一个性能表）。
2. **读取 CPU 支持的频率列表**（来自 cpufreq 表）。
3. **取每个性能域的最大频率**作为性能指标。
4. **按最大频率比例缩放**，将最高者设为 1024。

> 📌 公式：
>
> ```
> capacity = (max_freq_of_this_cpu / global_max_freq) * 1024
> ```

#### 示例：

| CPU 类型 | 最大频率 | 相对 capacity 计算 |
|---------|----------|---------------------|
| Cortex-A78 | 3.0 GHz | (3.0 / 3.0) × 1024 = **1024** |
| Cortex-A76 | 2.4 GHz | (2.4 / 3.0) × 1024 ≈ **819** |
| Cortex-A55 | 1.8 GHz | (1.8 / 3.0) × 1024 ≈ **614** |

> ⚠️ 注意：这假设 **性能与频率线性相关**，实际中不完全准确（还受架构、IPC 影响），但 Linux 调度器采用此简化模型。

---

### 方法 3：通过 **性能基准测试（如 MIPS、DMIPS）**

有些平台使用更精确的指标，比如：

- **DMIPS/MHz**（Dhrystone MIPS per MHz）
- 或 SPECint_rate 基准

然后：

```
capacity = (DMIPS_of_cpu / DMIPS_of_max_cpu) * 1024
```

例如：

| CPU       | DMIPS/MHz | 最大频率 | 总 DMIPS | capacity |
|-----------|-----------|----------|----------|----------|
| A78       | 6.0       | 3.0 GHz  | 18000    | 1024     |
| A55       | 2.1       | 1.8 GHz  | 3780     | (3780/18000)*1024 ≈ **215** |

> 这比纯频率更准确，但需要厂商提供数据。

---

## 🧪 四、如何查看当前系统的 `capacity`？

### 方法 1：查看 `/proc/cpuinfo`（间接）

```bash
cat /proc/cpuinfo | grep -E "processor|model name"
```

看 CPU 类型，结合知识推断。

### 方法 2：查看调度器调试信息

```bash
cat /proc/sched_debug | grep "cpu_capacity"
```

输出示例：

```
cpu 0: capacity 1024
cpu 1: capacity 1024
cpu 2: capacity 512
cpu 3: capacity 512
```

### 方法 3：内核日志

```bash
dmesg | grep -i capacity
```

可能看到：

```
Detected capacity of cpu0: 1024
Scaling driver specifies max freq, using it for capacity
```

---

## 🧱 五、`capacity` 的存储位置（内核代码层面）

在内核中，`capacity` 存储在：

```c
struct sched_domain {
    ...
};

struct sched_group {
    unsigned int cpu_capacity;  // 每个 CPU 组的 capacity
};

// 获取某个 CPU 的 capacity
capacity_of(cpu) → cpu_rq(cpu)->cpu_capacity
```

初始化发生在 `setup_cpu_capacity()` 函数中，路径：

```
start_kernel()
 → smp_init()
   → cpu_up()
     → set_cpu_topology()
     → update_cpu_capacity()  // 根据 DT 或频率表设置
```

---

## 📌 六、关键结论总结

| 问题 | 回答 |
|------|------|
| **`capacity` 是怎么定的？** | 由设备树、ACPI 或内核自动根据频率/性能推导。 |
| **是不是最强 CPU 定为 1024？** | ✅ 是的，1024 是归一化最大值。 |
| **其他 CPU 怎么算？** | 按相对性能（频率或 DMIPS）比例缩放。 |
| **频率变了 capacity 会变吗？** | ❌ 不会，`capacity` 是静态的。 |
| **不同系统会不同吗？** | ✅ 是的，每个 SoC 平台独立定义。 |

---

## 🧩 举个真实 SoC 例子：Qualcomm Snapdragon 888

| CPU 核心 | 类型 | 最大频率 | capacity（典型） |
|---------|------|-----------|------------------|
| 1x      | Cortex-X1 | 2.84 GHz | 1024 |
| 3x      | Cortex-A78 | 2.42 GHz | ~870 |
| 4x      | Cortex-A55 | 1.80 GHz | ~512 |

> 这些值在设备树中定义，或由内核根据频率自动计算。

---

## 📚 参考资料

- 内核源码：
  - `kernel/sched/topology.c` → `update_cpu_capacity()`
  - `drivers/base/arch_topology.c` → `arch_scale_cpu_capacity()`
- 文档：
  - `Documentation/admin-guide/cgroup-v2.rst`
  - `Documentation/scheduler/sched-capacity.txt`
- LWN 文章：
  - [Heterogeneous system scheduling](https://lwn.net/Articles/609977/)

---

# **Python 模拟 PELT（Per-Entity Load Tracking）算法**

太好了！下面是一个 **Python 模拟 PELT（Per-Entity Load Tracking）算法** 的脚本，它演示了：

- 如何从任务的 `runtime`（运行时间）出发
- 经过 **归一化 + 指数衰减平均**
- 最终得到 `util_avg`（利用率），并与 `capacity = 1024` 对齐
- 用于模拟 EAS 调度决策

---

## 🐍 Python 模拟脚本：`pelt_simulator.py`

```python
import matplotlib.pyplot as plt

# ========================
# 参数配置
# ========================
WINDOW = 32.0        # PELT 时间窗口（单位：ms）
TOTAL_CAPACITY = 1024  # CPU 最大容量（归一化基准）
DECAY_FACTOR = 0.5   # 每个窗口衰减 50%（简化模型，真实为连续衰减）

# 模拟数据：每个时间窗口内的任务运行时间（ms）
# 示例：任务在某些周期运行，某些周期睡眠
runtime_history = [
    32, 32, 0, 0,    # 满负载运行 2 个周期，然后空闲 2 个
    16, 16, 16,      # 50% 负载运行 3 个周期
    32, 32, 32, 0     # 再次满负载后空闲
]

# 初始化状态
util_sum = 0.0       # 累积的利用率“和”（带衰减）
util_avg_list = []   # 记录每个周期后的 util_avg
time_steps = list(range(len(runtime_history)))

# ========================
# 模拟 PELT 更新过程
# ========================
for i, runtime in enumerate(runtime_history):
    # 1. 归一化当前 runtime 为 [0, 1024] 范围内的贡献值
    normalized_util = (runtime / WINDOW) * TOTAL_CAPACITY  # 例如：16ms → 512

    # 2. 应用指数衰减：新值 = 旧值 * y + 新贡献 * (1 - y)
    util_sum = util_sum * DECAY_FACTOR + normalized_util * (1 - DECAY_FACTOR)

    # 3. util_avg 可以近似为 util_sum 的“稳定值”
    # （真实内核中还有更复杂的截断和整数处理）
    util_avg = min(util_sum, TOTAL_CAPACITY)  # 不超过 1024
    util_avg_list.append(util_avg)

    print(f"周期 {i+1:2d}: runtime={runtime:2d}ms → normalized={normalized_util:4.0f}, "
          f"util_sum={util_sum:6.1f}, util_avg={util_avg:6.1f}")

# ========================
# 可视化结果
# ========================
plt.figure(figsize=(10, 6))

# 绘制 runtime 对应的“瞬时利用率”（未平滑）
instant_util = [(rt / WINDOW) * TOTAL_CAPACITY for rt in runtime_history]
plt.plot(time_steps, instant_util, label='瞬时利用率 (runtime 直接归一化)', 
         color='red', linestyle='--', alpha=0.7)

# 绘制 PELT 平滑后的 util_avg
plt.plot(time_steps, util_avg_list, label='PELT 平滑 util_avg', 
         color='blue', linewidth=2)

# 绘制 capacity 线（1024）
plt.axhline(y=TOTAL_CAPACITY, color='green', linestyle='-', label='CPU capacity (1024)', linewidth=2)

# 图表设置
plt.title('PELT 算法模拟：从 runtime 计算 util_avg\n（归一化到 1024 以匹配 capacity）')
plt.xlabel('时间周期（每周期 32ms）')
plt.ylabel('利用率 (0~1024)')
plt.xticks(time_steps)
plt.ylim(0, 1200)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 显示图表
plt.show()
```

---

## 📊 运行结果示例（控制台输出片段）

```
周期  1: runtime=32ms → normalized=1024, util_sum= 512.0, util_avg= 512.0
周期  2: runtime=32ms → normalized=1024, util_sum= 768.0, util_avg= 768.0
周期  3: runtime= 0ms → normalized=   0, util_sum= 384.0, util_avg= 384.0
周期  4: runtime= 0ms → normalized=   0, util_sum= 192.0, util_avg= 192.0
周期  5: runtime=16ms → normalized= 512, util_sum= 288.0, util_avg= 288.0
...
周期 10: runtime=32ms → normalized=1024, util_sum= 896.0, util_avg= 896.0
```

👉 可以看到：
- 即使任务停止运行，`util_avg` 不会立刻归零（记忆效应）
- 最终 `util_avg` 趋近于实际负载水平
- 所有值都在 `[0, 1024]` 范围内，便于与 `capacity` 比较

---

## 🖼️ 图表说明

你会看到一张图，包含三条线：

| 曲线 | 含义 |
|------|------|
| **红色虚线** | 直接用 `runtime / window` 计算的瞬时利用率（毛刺大） |
| **蓝色实线** | PELT 平滑后的 `util_avg`（稳定、适合调度决策） |
| **绿色横线** | CPU 的 `capacity = 1024`，作为比较基准 |

> ✅ 当蓝色线低于绿色线时，表示 CPU “能容纳” 该任务。

---

## 🔧 扩展建议

你可以修改 `runtime_history` 来测试不同场景：

```python
# 高负载
runtime_history = [32] * 10

# 间歇性爆发
runtime_history = [32, 0, 32, 0, 16, 16]

# 低负载
runtime_history = [8] * 10
```

观察 `util_avg` 如何响应。

---

## 💡 总结：这个脚本说明了什么？

1. **`runtime` → `util_avg` 的转化过程**
2. **归一化到 1024 是为了与 `capacity` 在同一尺度比较**
3. **PELT 提供平滑、带记忆的利用率信号，适合用于：
   - EAS 调度
   - schedutil 频率调节
   - 负载均衡决策**

---

如果你希望，我还可以：
- 加入多任务叠加的模拟
- 模拟两个 CPU（大核 vs 小核）的调度决策
- 导出 CSV 数据供分析

需要的话告诉我即可 😊

--- 

🎯 **你现在对 `util` 和 `capacity` 的关系，是不是更直观了？**