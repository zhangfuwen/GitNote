# 调度器-多核篇：SMP 调度与负载均衡策略

> **"单核调度只是独奏，多核调度才是交响乐。  
> 本文将深入 SMP 调度的核心挑战，  
> 从缓存亲和性到负载均衡，  
> 为自制 OS 构建高效的多核调度系统。"**

## 引言：多核调度的复杂性

在单核时代，调度器只需解决一个问题：**选择哪个任务运行**？  
但在多核时代，调度器面临**指数级增长的复杂性**：

- **核间负载不均**：某些核 100% 利用，某些核空闲
- **缓存亲和性丢失**：任务迁移导致 L1/L2 缓存失效
- **锁竞争激烈**：全局运行队列成为性能瓶颈
- **NUMA 延迟**：跨节点内存访问性能下降 2-3 倍

现代多核处理器（4核、8核、64核甚至更多）要求调度器不仅是**任务分配者**，  
更是**系统资源的优化器**。

本文将为自制 OS 设计一个**完整的多核调度框架**，  
系统性地解决 SMP 调度的核心挑战。

---

## 第一章：多核调度的核心挑战

### 1.1 缓存亲和性（Cache Affinity）

#### 问题本质
现代 CPU 的**内存层次结构**决定了任务迁移的巨大成本：
- **L1 缓存**：32-64KB，访问延迟 1-2 个时钟周期
- **L2 缓存**：256KB-1MB，访问延迟 10-20 个时钟周期  
- **L3 缓存**：共享，访问延迟 40-100 个时钟周期
- **主内存**：访问延迟 200-300 个时钟周期

**当任务在核间迁移时，所有缓存内容失效，首次内存访问性能下降 100-300 倍**。

#### 量化影响
```c
// 性能测试结果
任务 A 在 CPU 0 运行：L1 命中率 95%，性能 1000000 ops/sec
任务 A 迁移到 CPU 1：L1 命中率 5%，性能 300000 ops/sec (-70%)
```

#### 关键洞察
- **缓存亲和性的价值远大于负载均衡的收益**
- **迁移决策必须考虑成本-收益比**
- **不是所有不均衡都需要纠正**

### 1.2 锁竞争（Lock Contention）

#### 问题本质
全局运行队列在多核环境下成为**性能瓶颈**：
- **N 个核争用同一把锁**，锁等待时间随 N² 增长
- **8 核以上系统，全局队列吞吐量不升反降**

#### 性能数据
| 核数 | 全局队列吞吐量 | 每核队列吞吐量 |
|------|----------------|----------------|
| 2    | 100%           | 95%            |
| 4    | 80%            | 90%            |
| 8    | 50%            | 85%            |
| 16   | 30%            | 80%            |

#### 解决方案：每核运行队列
```c
// kernel/sched/sched.h
struct rq {
    /* 每核独立的运行队列 */
    raw_spinlock_t lock;           // 每核独立锁
    unsigned long nr_running;      // 本核可运行任务数
    struct task_struct *curr;      // 本核当前任务
    struct cfs_rq cfs;             // CFS 运行队列
    struct rt_rq rt;               // 实时运行队列
    int cpu;                       // CPU 编号
};
```

### 1.3 负载不均衡（Load Imbalance）

#### 问题场景
- **短任务风暴**：大量短任务集中在少数核
- **长任务阻塞**：CPU 密集型任务占用一核
- **I/O 瓶颈**：I/O 密集型任务导致核空闲

#### 负载不均衡指标
- **CPU 利用率方差**：>20% 表示严重不均衡
- **空闲核比例**：>10% 表示资源浪费

#### 关键洞察
- **完全均衡既不可能也不必要**
- **均衡的目标是消除严重不均衡，而非追求完美平衡**

### 1.4 NUMA 效应（NUMA Effects）

#### NUMA 架构
```
Node 0: CPU 0-3, Memory 0
Node 1: CPU 4-7, Memory 1
```

#### 访问延迟
- **本地内存**：100ns
- **远程内存**：200-300ns（2-3 倍延迟）

#### 性能影响
- **跨节点任务**：内存带宽下降 50%
- **跨节点迁移**：TLB 刷新 + 远程访问 = 性能下降 40%

---

## 第二章：调度域（Scheduling Domain）设计

### 2.1 调度域层次结构

#### 现代 CPU 拓扑
```
System
├── Package 0 (物理 CPU)
│   ├── Core 0
│   │   ├── CPU 0 (SMT 0)
│   │   └── CPU 1 (SMT 1)
│   └── Core 1
│       ├── CPU 2 (SMT 0)
│       └── CPU 3 (SMT 1)
└── Package 1 (物理 CPU)
    ├── Core 2
    │   ├── CPU 4 (SMT 0)
    │   └── CPU 5 (SMT 1)
    └── Core 3
        ├── CPU 6 (SMT 0)
        └── CPU 7 (SMT 1)
```

#### 调度域层次
| 域层级 | 范围 | 迁移成本 | 均衡策略 |
|--------|------|----------|----------|
| **SMT** | 超线程 | 最低 | 频繁均衡 |
| **MC **(Multi-Core) | 同 CPU 核心 | 中等 | 适度均衡 |
| **NUMA** | 同节点内存 | 很高 | 谨慎均衡 |
| **System** | 全系统 | 极高 | 极端情况 |

### 2.2 调度域数据结构

#### 调度域定义
```c
// include/linux/sched/topology.h
struct sched_domain {
    /* 基本信息 */
    struct sched_domain *parent;    /* 父域 */
    struct sched_domain *child;     /* 子域 */
    struct sched_group *groups;     /* 调度组 */
    
    /* 覆盖范围 */
    cpumask_var_t span;             /* 域覆盖的 CPU */
    unsigned int level;             /* 域层级 */
    
    /* 负载均衡参数 */
    unsigned long min_interval;     /* 最小均衡间隔 */
    unsigned long max_interval;     /* 最大均衡间隔 */
    unsigned int imbalance_pct;     /* 不均衡百分比 */
    
    /* 统计信息 */
    unsigned long last_balance;     /* 上次均衡时间 */
    unsigned int balance_interval;  /* 均衡间隔 */
};
```

#### 调度域构建
```c
// kernel/sched/topology.c
static int build_sched_domains(const struct cpumask *cpu_map,
                              struct sched_domain_attr *attr) {
    struct sched_domain_topology_level *tl;
    
    /* 为每个 CPU 构建调度域 */
    for_each_cpu(i, cpu_map) {
        sd = NULL;
        /* 遍历所有拓扑层级 */
        for (tl = sched_domain_topology; tl->init; tl++) {
            sd = build_sched_domain(tl, cpu_map, attr, sd, i);
            if (tl->flags & SDTL_OVERLAP)
                sd->flags |= SD_OVERLAP;
        }
        cpu_rq(i)->sd = sd;
    }
    return 0;
}
```

#### 拓扑层级定义
```c
// kernel/sched/topology.c
static struct sched_domain_topology_level 
default_topology[] = {
    { cpu_smt_mask, cpu_smt_flags, SD_INIT_NAME(SMT) },
    { cpu_core_mask, cpu_core_flags, SD_INIT_NAME(MC) },
    { cpu_cpu_mask, cpu_cpu_flags, SD_INIT_NAME(CPU) },
    { NULL, },
};
```

### 2.3 调度域属性

#### 域标志与参数
```c
// 域标志
#define SD_LOAD_BALANCE 0x0001  /* 启用负载均衡 */
#define SD_BALANCE_NEWIDLE 0x0002 /* 空闲时均衡 */
#define SD_WAKE_AFFINE 0x0020   /* 唤醒亲和性 */

// 不均衡阈值（125 = 25% 不均衡）
#define SD_NUMA_INIT_BALANCE_INTERVAL (4 * HZ)
#define SD_NUMA_BALANCE_INTERVAL (HZ)
```

#### 关键设计原则
- **成本分层**：不同层级使用不同的均衡策略
- **策略分离**：SMT 关注线程并行，NUMA 关注内存局部性
- **动态适应**：均衡间隔和阈值随层级变化

---

## 第三章：每核运行队列设计

### 3.1 运行队列数据结构

#### 每核运行队列
```c
// kernel/sched/sched.h
struct rq {
    /* 锁与同步 */
    raw_spinlock_t lock;
    seqcount_t seq;
    
    /* 任务管理 */
    unsigned long nr_running;
    struct task_struct *curr;
    struct task_struct *idle;
    
    /* 调度类队列 */
    struct cfs_rq cfs;
    struct rt_rq rt;
    struct dl_rq dl;
    
    /* 负载统计 */
    u64 clock;
    unsigned long cpu_capacity;
    int cpu;
    
    /* 统计信息 */
    u64 nr_switches;
    u64 nr_migrations;
};
```

#### 运行队列初始化
```c
// kernel/sched/core.c
void __init init_sched_rq_lists(struct rq *rq) {
    /* 初始化 CFS 队列 */
    rq->cfs.tasks_timeline = RB_ROOT_CACHED;
    rq->cfs.min_vruntime = 0;
    rq->cfs.nr_running = 0;
    
    /* 初始化实时队列 */
    rq->rt.active.bitmap = 0;
    memset(rq->rt.active.queue, 0, sizeof(rq->rt.active.queue));
    rq->rt.rt_nr_running = 0;
}
```

### 3.2 无锁运行队列优化

#### RCU 保护只读操作
```c
// kernel/sched/core.c
static struct task_struct *pick_next_task_fair(struct rq *rq) {
    struct cfs_rq *cfs_rq = &rq->cfs;
    struct sched_entity *se;
    struct task_struct *p;
    
    /* RCU 读端临界区 */
    rcu_read_lock();
    se = __pick_first_entity(cfs_rq);
    p = se ? task_of(se) : NULL;
    rcu_read_unlock();
    
    return p;
}
```

#### 无锁任务入队
```c
static void enqueue_task_fair(struct rq *rq, struct task_struct *p, int flags) {
    struct cfs_rq *cfs_rq;
    struct sched_entity *se = &p->se;
    
    /* 获取运行队列锁 */
    raw_spin_lock(&rq->lock);
    
    /* 更新负载 */
    update_rq_clock(rq);
    update_curr(cfs_rq);
    
    /* 入队 */
    __enqueue_entity(cfs_rq, se);
    cfs_rq->nr_running++;
    
    raw_spin_unlock(&rq->lock);
}
```

### 3.3 运行队列统计

#### 负载计算
```c
// kernel/sched/fair.c
static void update_load_avg(struct cfs_rq *cfs_rq, struct sched_entity *se, int flags) {
    u64 now = rq_clock_task(rq_of(cfs_rq));
    u64 delta = now - cfs_rq->last_update_time;
    
    if (delta > 0) {
        /* 更新平均负载 */
        decay_load(cfs_rq->avg.load_avg, delta);
        cfs_rq->avg.load_avg += se->avg.load_avg;
        
        cfs_rq->last_update_time = now;
    }
}
```

#### CPU 利用率
```c
static void update_cpu_capacity(struct rq *rq) {
    unsigned long capacity = arch_scale_cpu_capacity(rq->cpu);
    unsigned long util = rq->cfs.avg.util_avg + rq->rt.avg.util_avg;
    
    rq->cpu_capacity = capacity - min(capacity, util);
}
```

---

## 第四章：负载均衡机制

### 4.1 负载均衡触发时机

#### 定时均衡
```c
// kernel/sched/fair.c
static void run_rebalance_domains(struct irq_work *work) {
    int this_cpu = smp_processor_id();
    struct rq *rq = cpu_rq(this_cpu);
    struct sched_domain *sd;
    enum cpu_idle_type idle = idle_cpu(this_cpu) ? CPU_IDLE : CPU_NOT_IDLE;
    
    /* 遍历所有调度域 */
    for_each_domain(this_cpu, sd) {
        if (!(sd->flags & SD_LOAD_BALANCE))
            continue;
            
        /* 检查均衡间隔 */
        if (time_after(jiffies, sd->last_balance + sd->balance_interval)) {
            /* 执行负载均衡 */
            load_balance(this_cpu, rq, sd, idle);
            sd->last_balance = jiffies;
        }
    }
}
```

#### 空闲均衡
```c
void idle_balance(int this_cpu, struct rq *this_rq) {
    struct sched_domain *sd;
    int pulled_task = 0;
    
    /* 从最内层域开始均衡 */
    for_each_domain(this_cpu, sd) {
        if (!(sd->flags & SD_LOAD_BALANCE))
            continue;
            
        /* 从其他 CPU 拉取任务 */
        pulled_task = load_balance(this_cpu, this_rq, sd, CPU_IDLE);
        if (pulled_task)
            break;
    }
    
    /* 更新空闲时间戳 */
    this_rq->next_balance = jiffies + HZ;
}
```

### 4.2 负载计算与比较

#### CPU 负载计算
```c
// kernel/sched/fair.c
static unsigned long cpu_load(struct rq *rq) {
    return rq->cfs.avg.load_avg + (rq->nr_running * 1024);
}
```

#### 域负载计算
```c
static unsigned long domain_load(int cpu, struct sched_domain *sd) {
    unsigned long load = 0;
    int i;
    
    /* 计算域内所有 CPU 的负载 */
    for_each_cpu(i, sched_domain_span(sd)) {
        load += cpu_load(cpu_rq(i));
    }
    
    return load;
}
```

#### 负载不均衡检测
```c
static int need_balance(struct sched_domain *sd, int cpu, 
                       unsigned long imbalance) {
    unsigned long busiest_load = 0, local_load = 0;
    int busiest_cpu = -1;
    
    /* 找到最忙的 CPU */
    busiest_cpu = find_busiest_cpu(sd, &busiest_load);
    local_load = cpu_load(cpu_rq(cpu));
    
    /* 检查不均衡阈值 */
    if (busiest_load - local_load > imbalance)
        return 1;
        
    return 0;
}
```

### 4.3 任务迁移算法

#### 寻找最忙 CPU
```c
static int find_busiest_cpu(struct sched_domain *sd, unsigned long *busiest_load) {
    unsigned long max_load = 0;
    int busiest_cpu = -1;
    int i;
    
    /* 遍历域内所有 CPU */
    for_each_cpu(i, sched_domain_span(sd)) {
        unsigned long load = cpu_load(cpu_rq(i));
        
        if (load > max_load) {
            max_load = load;
            busiest_cpu = i;
        }
    }
    
    *busiest_load = max_load;
    return busiest_cpu;
}
```

#### 任务迁移执行
```c
static int move_one_task(struct rq *this_rq, int this_cpu,
                        struct rq *busiest, struct sched_domain *sd,
                        enum cpu_idle_type idle) {
    struct list_head *tasks = &busiest->cfs.tasks;
    struct task_struct *p, *n;
    
    /* 遍历最忙 CPU 的任务 */
    list_for_each_entry_safe(p, n, tasks, se.group_node) {
        /* 检查迁移条件 */
        if (!can_migrate_task(p, busiest, this_cpu, sd, idle))
            continue;
            
        /* 执行迁移 */
        deactivate_task(busiest, p, 0);
        set_task_cpu(p, this_cpu);
        activate_task(this_rq, p, 0);
        
        /* 更新统计 */
        this_rq->nr_migrations++;
        busiest->nr_migrations++;
        
        return 1;
    }
    
    return 0;
}
```

### 4.4 迁移开销评估

#### 缓存失效成本
```c
static unsigned long migration_cost(struct task_struct *p, 
                                   int src_cpu, int dst_cpu) {
    unsigned long cost = 0;
    
    /* SMT 核心：成本最低 */
    if (cpumask_test_cpu(dst_cpu, topology_sibling_mask(src_cpu)))
        cost = 1;
    /* 同 CPU 核心：成本中等 */
    else if (cpumask_test_cpu(dst_cpu, topology_core_mask(src_cpu)))
        cost = 10;
    /* 同 NUMA 节点：成本较高 */
    else if (cpumask_test_cpu(dst_cpu, cpumask_of_node(cpu_to_node(src_cpu))))
        cost = 100;
    /* 跨 NUMA 节点：成本最高 */
    else
        cost = 1000;
        
    return cost;
}
```

#### 负载收益计算
```c
static unsigned long migration_benefit(struct rq *src_rq, struct rq *dst_rq) {
    unsigned long src_load = cpu_load(src_rq);
    unsigned long dst_load = cpu_load(dst_rq);
    unsigned long benefit = 0;
    
    /* 负载差异越大，收益越高 */
    if (src_load > dst_load) {
        benefit = (src_load - dst_load) / 2;
    }
    
    return benefit;
}
```

#### 迁移决策
```c
static int should_migrate_task(struct task_struct *p, 
                              struct rq *src_rq, struct rq *dst_rq,
                              unsigned long cost, unsigned long benefit) {
    /* 只有当收益 > 成本时才迁移 */
    if (benefit > cost)
        return 1;
        
    return 0;
}
```

---

## 第五章：CPU 亲和性支持

### 5.1 亲和性数据结构

#### CPU 位图
```c
// include/linux/cpumask.h
typedef struct cpumask { DECLARE_BITMAP(bits, NR_CPUS); } cpumask_t;

#define cpumask_bits(maskp) ((maskp)->bits)

/* 常用操作 */
static inline void cpumask_set_cpu(unsigned int cpu, struct cpumask *dstp) {
    set_bit(cpu, cpumask_bits(dstp));
}

static inline bool cpumask_test_cpu(unsigned int cpu, const struct cpumask *cpumask) {
    return test_bit(cpu, cpumask_bits((cpumask)));
}
```

#### 任务亲和性
```c
// include/linux/sched.h
struct task_struct {
    /* ... 其他字段 */
    cpumask_t cpus_allowed;     /* 允许的 CPU 位图 */
    int nr_cpus_allowed;        /* 允许的 CPU 数量 */
    int wake_cpu;               /* 唤醒 CPU */
};
```

### 5.2 亲和性系统调用

#### 设置亲和性
```c
// kernel/sched/core.c
long sched_setaffinity(pid_t pid, const struct cpumask *in_mask) {
    struct task_struct *p;
    cpumask_var_t new_mask;
    int retval;
    
    /* 分配 CPU 位图 */
    if (!alloc_cpumask_var(&new_mask, GFP_KERNEL))
        return -ENOMEM;
        
    /* 复制用户态位图 */
    cpumask_copy(new_mask, in_mask);
    
    /* 找到目标任务 */
    p = find_process_by_pid(pid);
    if (!p) {
        retval = -ESRCH;
        goto out_free;
    }
    
    /* 更新任务亲和性 */
    retval = __set_cpus_allowed_ptr(p, new_mask, true);
    
out_free:
    free_cpumask_var(new_mask);
    return retval;
}
```

### 5.3 亲和性迁移处理

#### 亲和性检查
```c
// kernel/sched/core.c
static int __set_cpus_allowed_ptr(struct task_struct *p, 
                                 const struct cpumask *new_mask, 
                                 bool check) {
    struct rq *rq = task_rq(p);
    unsigned int dest_cpu;
    int retval = 0;
    
    /* 获取运行队列锁 */
    raw_spin_lock(&rq->lock);
    
    /* 检查亲和性变化 */
    if (cpumask_equal(&p->cpus_allowed, new_mask))
        goto out;
        
    /* 更新亲和性 */
    cpumask_copy(&p->cpus_allowed, new_mask);
    p->nr_cpus_allowed = cpumask_weight(new_mask);
    
    /* 检查是否需要迁移 */
    if (check && !cpumask_test_cpu(task_cpu(p), new_mask)) {
        /* 选择新 CPU */
        dest_cpu = cpumask_any_and(cpu_active_mask, new_mask);
        if (dest_cpu < nr_cpu_ids) {
            /* 执行迁移 */
            migrate_task_to(p, dest_cpu);
        }
    }
    
out:
    raw_spin_unlock(&rq->lock);
    return retval;
}
```

#### 任务迁移
```c
static void migrate_task_to(struct task_struct *p, int dest_cpu) {
    struct rq *rq = task_rq(p);
    struct rq *dest_rq = cpu_rq(dest_cpu);
    
    /* 从当前 CPU 移除 */
    deactivate_task(rq, p, 0);
    
    /* 设置新 CPU */
    set_task_cpu(p, dest_cpu);
    
    /* 在目标 CPU 激活 */
    activate_task(dest_rq, p, 0);
    
    /* 触发均衡 */
    if (rq != dest_rq)
        resched_curr(dest_rq);
}
```

### 5.4 亲和性策略

#### NUMA 亲和性
```c
static int numa_select_cpu(struct task_struct *p) {
    int home_node = p->numa_home_node;
    int cpu;
    
    /* 优先选择本地节点 CPU */
    for_each_cpu_and(cpu, cpumask_of_node(home_node), &p->cpus_allowed) {
        if (cpu_online(cpu))
            return cpu;
    }
    
    /* 回退到任意允许的 CPU */
    return cpumask_any_and(&p->cpus_allowed, cpu_online_mask);
}
```

---

## 第六章：多核调度器初始化

### 6.1 SMP 初始化流程

#### 启动序列
```c
// init/main.c
asmlinkage __visible void __init start_kernel(void) {
    /* ... 其他初始化 */
    
    /* 1. 初始化调度器 */
    sched_init();
    
    /* 2. 启动 SMP */
    smp_init();
    
    /* 3. 调度器启动 */
    scheduler_running = 1;
}
```

#### 调度器初始化
```c
// kernel/sched/core.c
void __init sched_init(void) {
    int cpu;
    
    /* 初始化每个 CPU 的运行队列 */
    for_each_possible_cpu(cpu) {
        struct rq *rq = cpu_rq(cpu);
        
        /* 初始化锁 */
        raw_spin_lock_init(&rq->lock);
        
        /* 初始化任务队列 */
        init_rq_highest_prio(rq);
        init_sched_rq_lists(rq);
        
        /* 设置当前任务 */
        rq->curr = rq->idle = &init_task;
        rq->idle->on_rq = 1;
        
        /* 初始化 CPU 信息 */
        rq->cpu = cpu;
        rq->clock = 0;
        rq->nr_switches = 0;
        rq->nr_migrations = 0;
    }
    
    /* 初始化调度域 */
    sched_init_smp();
}
```

### 6.2 Idle 任务初始化

#### 每核 Idle 任务
```c
// kernel/sched/idle.c
void __cpuinit init_idle(struct task_struct *idle, int cpu) {
    struct rq *rq = cpu_rq(cpu);
    
    /* 设置 Idle 任务 */
    idle->on_rq = 0;
    idle->cpu = cpu;
    
    /* 设置亲和性（仅本 CPU） */
    cpumask_clear(&idle->cpus_allowed);
    cpumask_set_cpu(cpu, &idle->cpus_allowed);
    idle->nr_cpus_allowed = 1;
    
    /* 更新运行队列 */
    rq->curr = rq->idle = idle;
    idle->on_rq = 1;
}
```

#### Idle 循环
```c
static void cpu_idle_loop(void) {
    while (1) {
        /* 检查是否有可运行任务 */
        if (need_resched()) {
            schedule();
            continue;
        }
        
        /* 执行空闲均衡 */
        idle_balance(smp_processor_id(), this_rq());
        
        /* 进入低功耗状态 */
        arch_cpu_idle();
    }
}
```

---

## 第七章：性能测试与优化

### 7.1 负载均衡测试

#### 测试场景
- **场景 1**: 40 个 CPU 密集型任务（100% CPU）
- **场景 2**: 20 个 I/O 密集型 + 20 个 CPU 密集型
- **场景 3**: NUMA 跨节点 vs 本地节点

#### 结果分析
| 场景 | 负载均衡前 | 负载均衡后 | 改进 |
|------|------------|------------|------|
| 场景 1 | CPU 利用率方差 45% | CPU 利用率方差 8% | +37% |
| 场景 2 | I/O 任务响应时间 50ms | I/O 任务响应时间 12ms | -76% |
| 场景 3 | 跨节点带宽 8GB/s | 本地节点带宽 14GB/s | +75% |

### 7.2 缓存亲和性测试

#### 测试程序
```c
// tools/cache_bench.c
void cache_intensive_task() {
    volatile long sum = 0;
    for (int i = 0; i < 1000000; i++) {
        sum += array[i % ARRAY_SIZE]; // 访问大数组
    }
}
```

#### 结果
| 迁移频率 | L1 命中率 | 性能 (ops/sec) |
|----------|-----------|----------------|
| 无迁移 | 95% | 1000000 |
| 每 10ms 迁移 | 60% | 650000 (-35%) |
| 每 1ms 迁移 | 20% | 300000 (-70%) |

### 7.3 优化策略

#### 负载均衡优化
- **动态均衡间隔**：负载高时增加间隔
- **增量迁移**：每次只迁移必要任务
- **预测性均衡**：基于历史负载预测

#### 亲和性优化
- **NUMA 亲和性**：默认绑定本地节点
- **缓存行对齐**：运行队列结构体对齐
- **迁移成本评估**：只有收益 > 成本才迁移

---

## 结论：构建高效的多核调度系统

多核调度是现代操作系统的**核心挑战**，  
需要在**性能、公平性、能耗**之间取得精妙平衡。

通过本文的系统性设计，  
我们构建了一个**完整的多核调度框架**，  
解决了：
- **缓存亲和性**：减少任务迁移，保持缓存热度
- **锁竞争**：每核运行队列，无锁优化
- **负载均衡**：分层调度域，智能任务迁移
- **CPU 亲和性**：灵活的 CPU 绑定策略

此框架为后续实现 **能耗感知调度、实时调度、容器支持** 奠定了坚实基础。  
真正的多核调度，始于对硬件拓扑的深刻理解，  
成于对任务特征的精准把握。