# 调度器-高级篇：节能调度、NUMA 亲和性与容器支持

> **"现代调度器不仅是任务分配者，更是系统资源的智能优化器。  
> 本文将深入调度器的高级特性，  
> 从节能调度到 NUMA 亲和性，  
> 为自制 OS 构建企业级调度能力。"**

## 引言：调度器的企业级需求

在现代数据中心和云环境中，调度器面临前所未有的复杂需求：
- **能耗成本**：数据中心 30% 成本来自电力
- **NUMA 拓扑**：多路服务器的内存访问延迟差异
- **资源隔离**：容器和虚拟机的资源争用
- **性能分析**：复杂系统的调试和优化

传统的调度器（如 CFS）虽然公平高效，  
但缺乏对这些**企业级特性**的支持。

本文将深入现代调度器的高级特性，  
为自制 OS 构建完整的**企业级调度能力**。

---

## 第一章：节能调度

### 1.1 节能调度的核心挑战

#### 问题本质
现代 CPU 的**功耗与频率呈非线性关系**：
- **频率 f**：CPU 时钟频率
- **电压 V**：工作电压
- **功耗 P ∝ C × V² × f**

**降低频率可显著降低功耗，但影响性能**。

#### 调度器的权衡
- **高性能模式**：高频率，低延迟，高功耗
- **节能模式**：低频率，高延迟，低功耗
- **平衡模式**：动态调整，适应负载

### 1.2 CPU 频率调节

#### 频率调节接口
```c
// arch/x86/kernel/cpu/freq.c
void set_cpu_frequency(int cpu, unsigned long freq)
{
    struct cpufreq_policy *policy = cpufreq_cpu_get(cpu);
    
    if (policy) {
        /* 设置目标频率 */
        __cpufreq_driver_target(policy, freq, CPUFREQ_RELATION_H);
        cpufreq_cpu_put(policy);
    }
}
```

#### 调度器集成
```c
// kernel/sched/fair.c
static void energy_aware_task_tick(struct rq *rq, struct task_struct *p)
{
    unsigned long util = rq->cfs.avg.util_avg;
    unsigned long freq;
    
    /* 根据负载调整频率 */
    if (util > HIGH_LOAD_THRESHOLD) {
        freq = MAX_FREQUENCY;    /* 高负载：高性能 */
    } else if (util < LOW_LOAD_THRESHOLD) {
        freq = MIN_FREQUENCY;    /* 低负载：节能 */
    } else {
        freq = util * MAX_FREQUENCY / 100; /* 平衡 */
    }
    
    set_cpu_frequency(rq->cpu, freq);
}
```

### 1.3 核休眠（CPU Idle）

#### 空闲状态管理
```c
// kernel/sched/idle.c
void cpu_idle_loop(void)
{
    while (1) {
        /* 检查是否有可运行任务 */
        if (need_resched()) {
            schedule();
            continue;
        }
        
        /* 选择最优空闲状态 */
        int next_state = select_idle_state();
        
        /* 进入空闲状态 */
        cpuidle_enter(next_state);
    }
}
```

#### 空闲状态选择
```c
static int select_idle_state(void)
{
    u64 predicted_idle_time = predict_idle_duration();
    
    /* 根据预测空闲时间选择状态 */
    if (predicted_idle_time > 100000) {      /* >100ms */
        return CPUIDLE_STATE_DEEP;           /* 深度休眠 */
    } else if (predicted_idle_time > 10000) { /* >10ms */
        return CPUIDLE_STATE_MEDIUM;         /* 中度休眠 */
    } else {
        return CPUIDLE_STATE_LIGHT;          /* 轻度休眠 */
    }
}
```

### 1.4 节能调度策略

#### 调度类扩展
```c
// kernel/sched/energy.c
struct energy_sched_class {
    struct sched_class base;
    void (*energy_task_tick)(struct rq *rq, struct task_struct *p);
    int (*select_energy_cpu)(struct task_struct *p);
};

static struct energy_sched_class energy_sched = {
    .base = {
        .enqueue_task = enqueue_task_fair,
        .pick_next_task = pick_next_task_fair,
        .task_tick = energy_aware_task_tick,
    },
    .energy_task_tick = energy_aware_task_tick,
    .select_energy_cpu = select_energy_cpu,
};
```

#### 能效核调度
```c
static int select_energy_cpu(struct task_struct *p)
{
    int best_cpu = -1;
    unsigned long min_energy = ULONG_MAX;
    int cpu;
    
    /* 遍历所有允许的 CPU */
    for_each_cpu(cpu, &p->cpus_allowed) {
        if (!cpu_online(cpu))
            continue;
            
        /* 计算能耗 */
        unsigned long energy = calculate_cpu_energy(cpu, p);
        if (energy < min_energy) {
            min_energy = energy;
            best_cpu = cpu;
        }
    }
    
    return best_cpu;
}
```

---

## 第二章：NUMA 亲和性

### 2.1 NUMA 架构挑战

#### NUMA 拓扑
```
Node 0: CPU 0-3, Memory 0 (本地延迟 100ns)
Node 1: CPU 4-7, Memory 1 (本地延迟 100ns, 远程延迟 250ns)
```

#### 性能影响
- **本地内存访问**：100ns
- **远程内存访问**：250ns（2.5 倍延迟）
- **跨节点带宽**：通常为本地带宽的 50-70%

### 2.2 NUMA 拓扑检测

#### 拓扑信息获取
```c
// arch/x86/kernel/smpboot.c
static void detect_numa_topology(void)
{
    int cpu;
    
    for_each_possible_cpu(cpu) {
        struct cpuinfo_x86 *c = &cpu_data(cpu);
        
        /* 从 ACPI 或 CPUID 获取 NUMA 信息 */
        c->phys_proc_id = topology_physical_package_id(cpu);
        c->numa_node = cpu_to_node(cpu);
    }
}
```

#### NUMA 距离表
```c
// include/linux/numa.h
extern int node_distance(int a, int b);

static inline int node_distance(int a, int b)
{
    if (a == b)
        return LOCAL_DISTANCE;    /* 10 */
    else
        return REMOTE_DISTANCE;   /* 20 */
}
```

### 2.3 NUMA 亲和性调度

#### 任务放置策略
```c
// kernel/sched/fair.c
static int numa_select_cpu(struct task_struct *p)
{
    int home_node = p->numa_home_node;
    int cpu;
    
    /* 1. 优先选择本地节点 CPU */
    for_each_cpu_and(cpu, cpumask_of_node(home_node), &p->cpus_allowed) {
        if (cpu_online(cpu))
            return cpu;
    }
    
    /* 2. 回退到最近节点 */
    int closest_node = find_closest_node(home_node, &p->cpus_allowed);
    for_each_cpu_and(cpu, cpumask_of_node(closest_node), &p->cpus_allowed) {
        if (cpu_online(cpu))
            return cpu;
    }
    
    /* 3. 任意允许的 CPU */
    return cpumask_any_and(&p->cpus_allowed, cpu_online_mask);
}
```

#### 内存分配亲和性
```c
// mm/page_alloc.c
struct page *alloc_pages_node(int nid, gfp_t gfp_mask, unsigned int order)
{
    struct page *page;
    
    /* 1. 优先本地节点分配 */
    page = __alloc_pages_node(nid, gfp_mask, order);
    if (page)
        return page;
    
    /* 2. 回退到其他节点 */
    return __alloc_pages(gfp_mask, order, nid);
}
```

### 2.4 NUMA 负载均衡

#### NUMA 感知均衡
```c
// kernel/sched/fair.c
static int numa_balance(struct rq *this_rq, struct rq *busiest_rq)
{
    /* 1. 检查是否值得跨节点迁移 */
    if (node_distance(cpu_to_node(this_rq->cpu), 
                     cpu_to_node(busiest_rq->cpu)) > LOCAL_DISTANCE) {
        /* 跨节点迁移成本高，需更严格的条件 */
        if (busiest_rq->nr_running - this_rq->nr_running < 2)
            return 0;
    }
    
    /* 2. 执行迁移 */
    return move_one_task(this_rq, busiest_rq);
}
```

#### 迁移成本评估
```c
static unsigned long migration_cost(struct task_struct *p, int src_cpu, int dst_cpu)
{
    int src_node = cpu_to_node(src_cpu);
    int dst_node = cpu_to_node(dst_cpu);
    
    if (src_node == dst_node) {
        return 10;    /* 同节点：成本低 */
    } else {
        return 100;   /* 跨节点：成本高 */
    }
}
```

---

## 第三章：容器支持

### 3.1 cgroups 资源限制

#### cgroups 架构
```
/cgroup/cpu/
├── container1/
│   ├── cpu.cfs_quota_us    # CPU 配额
│   ├── cpu.cfs_period_us   # CPU 周期
│   └── tasks               # 任务列表
└── container2/
    ├── cpu.cfs_quota_us
    ├── cpu.cfs_period_us
    └── tasks
```

#### CPU 带宽控制
```c
// kernel/sched/cgroups.c
struct cfs_bandwidth {
    raw_spinlock_t lock;
    ktime_t period;            /* 周期 (100ms) */
    u64 quota;                 /* 配额 (50ms = 50% CPU) */
    u64 runtime;               /* 剩余运行时间 */
};

static void account_cfs_rq_runtime(struct cfs_rq *cfs_rq, u64 delta_exec)
{
    struct cfs_bandwidth *cfs_b = &cfs_rq->tg->cfs_bandwidth;
    
    /* 检查带宽限制 */
    if (cfs_b->quota != RUNTIME_INF) {
        cfs_rq->runtime_remaining -= delta_exec;
        if (cfs_rq->runtime_remaining <= 0) {
            /* 带宽用尽，节流任务 */
            throttle_cfs_rq(cfs_rq);
        }
    }
}
```

### 3.2 容器调度隔离

#### 任务组调度
```c
// kernel/sched/fair.c
struct task_group {
    struct cfs_rq **cfs_rq;        /* 每 CPU CFS 队列 */
    struct sched_entity **se;      /* 每 CPU 调度实体 */
    struct cfs_bandwidth cfs_bandwidth;
};

static void enqueue_task_fair(struct rq *rq, struct task_struct *p, int flags)
{
    struct sched_entity *se = &p->se;
    
    /* 从叶节点到根节点入队 */
    for_each_sched_entity(se) {
        struct cfs_rq *cfs_rq = cfs_rq_of(se);
        enqueue_entity(cfs_rq, se, flags);
    }
}
```

#### 层次化带宽分配
```c
static int tg_set_cfs_quota(struct task_group *tg, s64 quota)
{
    struct cfs_bandwidth *cfs_b = &tg->cfs_bandwidth;
    u64 period = ktime_to_ns(cfs_b->period);
    
    if (quota < 0) {
        cfs_b->quota = RUNTIME_INF;    /* 无限制 */
    } else {
        cfs_b->quota = quota;
        
        /* 检查层次配额 */
        if (tg->parent) {
            struct cfs_bandwidth *parent_b = &tg->parent->cfs_bandwidth;
            if (quota > parent_b->quota)
                return -EINVAL;    /* 子组配额不能超过父组 */
        }
    }
    
    return 0;
}
```

### 3.3 容器性能隔离

#### CPU 亲和性隔离
```c
// 设置容器 CPU 亲和性
echo 0-3 > /cgroup/cpuset/container1/cpuset.cpus
echo 0 > /cgroup/cpuset/container1/cpuset.mems
```

#### 内存带宽隔离
```c
// Intel RDT (Resource Director Technology)
echo "0=7f;1=7f" > /sys/fs/resctrl/schemata
echo $$ > /sys/fs/resctrl/container1/tasks
```

---

## 第四章：调度器调试与性能分析

### 4.1 调度器状态监控

#### /proc/sched_debug
```c
// kernel/sched/debug.c
static int sched_debug_show(struct seq_file *m, void *v)
{
    int cpu;
    
    for_each_online_cpu(cpu) {
        struct rq *rq = cpu_rq(cpu);
        
        seq_printf(m, "cpu#%d\n", cpu);
        seq_printf(m, "  .nr_running          : %lu\n", rq->nr_running);
        seq_printf(m, "  .load                : %lu\n", rq->load.weight);
        seq_printf(m, "  .cfs_rq[0]\n");
        seq_printf(m, "    .exec_clock        : %llu\n", rq->cfs.exec_clock);
        seq_printf(m, "    .min_vruntime      : %llu\n", rq->cfs.min_vruntime);
        seq_printf(m, "    .nr_running        : %lu\n", rq->cfs.nr_running);
    }
    
    return 0;
}
```

#### 任务级统计
```c
// /proc/[pid]/sched
se.exec_start                    :    123456789.123456
se.vruntime                      :    1000000000.000000
se.sum_exec_runtime              :     500000000.000000
nr_switches                      :                12345
```

### 4.2 性能分析工具

#### perf sched
```bash
# 记录调度事件
perf record -e sched:sched_wakeup,sched:sched_switch -a sleep 10

# 分析调度延迟
perf script -i perf.data | perf sched latency

# 输出示例
#  task             1:  1024.557 ms
#  task             2:   512.234 ms
```

#### 调度火焰图
```bash
# 生成调度火焰图
perf record -g -e sched:sched_switch -a sleep 30
perf script | ./stackcollapse-perf.pl | ./flamegraph.pl > sched.svg
```

### 4.3 调度器参数调优

#### 关键参数
| 参数 | 默认值 | 作用 | 调优建议 |
|------|--------|------|----------|
| **sched_migration_cost** | 500μs | 迁移成本阈值 | NUMA 系统：增大 |
| **sched_latency** | 6ms | 调度周期 | 交互式系统：减小 |
| **numa_balancing_scan_size** | 256MB | NUMA 扫描大小 | 大内存系统：增大 |

#### 动态调优
```c
// kernel/sched/tune.c
void dynamic_sched_tuning(void)
{
    unsigned long load = get_system_load();
    int numa_nodes = num_online_nodes();
    
    if (numa_nodes > 1) {
        /* NUMA 系统：增大迁移成本 */
        sysctl_sched_migration_cost = 2000;  /* 2ms */
    }
    
    if (load < 0.3) {
        /* 低负载：提高响应性 */
        sysctl_sched_latency = 3;
    } else if (load > 0.8) {
        /* 高负载：提高吞吐量 */
        sysctl_sched_latency = 12;
    }
}
```

---

## 第五章：sched_ext 高级用例

### 5.1 AI 驱动的调度策略

#### 机器学习调度器架构
```
+------------------+
|   特征提取器      |  // 从任务行为提取特征
+------------------+
|   在线推理引擎    |  // 实时预测最优调度
+------------------+
|   调度执行器      |  // 应用预测结果
+------------------+
```

#### 特征提取
```c
// sched_ext 程序
SEC("sched")
int ml_feature_extract(struct task_struct *task)
{
    struct task_features features = {};
    
    /* 1. CPU 使用率 */
    features.cpu_usage = task->se.sum_exec_runtime / (bpf_ktime_get_ns() - task->start_time);
    
    /* 2. I/O 行为 */
    features.io_ratio = task->io_wait_time / task->se.sum_exec_runtime;
    
    /* 3. 内存访问模式 */
    features.cache_miss_rate = get_cache_miss_rate(task);
    
    /* 4. 通信模式 */
    features.comm_frequency = get_communication_frequency(task);
    
    /* 存储特征 */
    bpf_map_update_elem(&task_features_map, &task->pid, &features, BPF_ANY);
    
    return 0;
}
```

#### 在线推理
```c
SEC("sched")
int ml_inference(struct sched_ext_run_ctx *ctx)
{
    struct task_struct *task = ctx->current;
    struct task_features *features;
    int predicted_priority;
    
    /* 1. 获取特征 */
    features = bpf_map_lookup_elem(&task_features_map, &task->pid);
    if (!features)
        return -1;
    
    /* 2. 简单规则推理 */
    if (features->io_ratio > 0.5 && features->cpu_usage < 0.1) {
        predicted_priority = INTERACTIVE_PRIORITY;    /* 交互式 */
    } else if (features->cpu_usage > 0.8) {
        predicted_priority = BATCH_PRIORITY;          /* 批处理 */
    } else {
        predicted_priority = NORMAL_PRIORITY;         /* 普通 */
    }
    
    /* 3. 应用预测结果 */
    task->priority = predicted_priority;
    
    return task->pid;
}
```

### 5.2 数据库专用调度器

#### 问题场景优化
```c
SEC("sched")
int db_scheduler(struct sched_ext_run_ctx *ctx)
{
    struct task_struct *task = ctx->current;
    u64 now = bpf_ktime_get_ns();
    
    /* 1. 识别短查询 */
    if (task->query_type == QUERY_SHORT && 
        task->expected_runtime < SHORT_QUERY_THRESHOLD) {
        /* 短查询：最高优先级，立即调度 */
        bpf_map_push_elem(&short_query_queue, task, BPF_ANY);
        return task->pid;
    }
    
    /* 2. 长查询并发控制 */
    if (task->query_type == QUERY_LONG) {
        if (bpf_map_len(&long_query_queue) < MAX_CONCURRENT_QUERIES) {
            bpf_map_push_elem(&long_query_queue, task, BPF_ANY);
            return task->pid;
        }
        /* 超过并发限制：暂时阻塞 */
        return -1;
    }
    
    return -1;
}
```

### 5.3 实时音视频调度器

#### 截止时间保证
```c
SEC("sched")
int av_scheduler(struct sched_ext_run_ctx *ctx)
{
    struct task_struct *task = ctx->current;
    u64 now = bpf_ktime_get_ns();
    
    /* 1. 音频任务：严格截止时间 */
    if (task->task_type == TASK_AUDIO) {
        if (now < task->deadline) {
            return task->pid;    /* 按时完成 */
        } else {
            /* 错过截止时间：丢弃或降级 */
            handle_missed_deadline(task);
            return -1;
        }
    }
    
    /* 2. 视频任务：帧率保证 */
    if (task->task_type == TASK_VIDEO) {
        if (now - task->last_frame_time > FRAME_INTERVAL) {
            return task->pid;    /* 需要新帧 */
        }
    }
    
    return -1;
}
```

---

## 第六章：企业级调度最佳实践

### 6.1 混合工作负载调度

#### 策略组合
- **实时任务**：使用 EDF 或 RMS
- **交互式任务**：使用 CFS + 低延迟优化
- **批处理任务**：使用 CFS + 高吞吐量优化
- **容器任务**：使用 cgroups 隔离

#### 优先级层次
```
1. 实时任务 (SCHED_FIFO/SCHED_RR)
2. 系统关键任务 (migration, ksoftirqd)
3. 交互式任务 (nice -20 ~ 0)
4. 普通任务 (nice 0)
5. 批处理任务 (nice 1 ~ 19)
6. 容器任务 (cgroups 限制)
```

### 6.2 NUMA 感知部署

#### 部署策略
- **数据库**：CPU 和内存绑定到同一 NUMA 节点
- **Web 服务器**：跨 NUMA 节点部署以利用更多资源
- **HPC 应用**：MPI 进程绑定到特定 NUMA 节点

#### 配置示例
```bash
# 数据库 NUMA 绑定
numactl --cpunodebind=0 --membind=0 /usr/bin/mysqld

# Web 服务器跨节点
numactl --interleave=all /usr/bin/nginx

# HPC 进程绑定
mpirun --bind-to core --map-by node ./mpi_app
```

### 6.3 节能与性能平衡

#### 动态策略
- **白天**：高性能模式，保证用户体验
- **夜间**：节能模式，降低运营成本
- **突发负载**：临时切换到高性能模式

#### 自动化脚本
```bash
#!/bin/bash
# 根据时间自动调整调度策略
HOUR=$(date +%H)

if [ $HOUR -ge 9 ] && [ $HOUR -le 18 ]; then
    # 工作时间：高性能
    echo 0 > /sys/devices/system/cpu/cpufreq/policy0/energy_performance_preference
else
    # 非工作时间：节能
    echo power > /sys/devices/system/cpu/cpufreq/policy0/energy_performance_preference
fi
```

---

## 结论：企业级调度的艺术

现代调度器已从简单的任务分配器，  
演变为**系统资源的智能优化器**。

通过本文的深入分析，  
我们构建了一个**完整的企业级调度框架**，  
涵盖了：
- **节能调度**：动态频率调节和核休眠
- **NUMA 亲和性**：本地内存优先和拓扑感知
- **容器支持**：cgroups 资源限制和隔离
- **性能分析**：调试工具和参数调优
- **可编程调度**：AI 驱动的高级用例

真正的企业级调度，  
不仅需要技术深度，  
更需要对**业务场景的深刻理解**。

调度的艺术，  
在于在**性能、能耗、成本、可靠性**之间找到最优平衡点。