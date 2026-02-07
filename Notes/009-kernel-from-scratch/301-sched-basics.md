# 调度器-基础篇：CPU 资源分配的核心问题与设计框架

> **“调度器不仅是任务的分配者，更是 CPU 核心的指挥官。  
> 本文将从零设计一个完整的调度框架，  
> 解决任务选择、CPU 选择、负载均衡三大核心问题，  
> 为多核时代的高效调度奠定基础。”**

## 引言：现代调度器的双重职责

在单核时代，调度器只需解决一个问题：**选择哪个任务运行**？  
但在多核时代，调度器面临**双重挑战**：
1. **任务选择**（Task Selection）：从可运行任务中选最优者
2. **CPU 选择**（CPU Selection）：决定任务在哪个核心上运行

更复杂的是，**负载均衡**（Load Balancing）成为必须：  
- **核间负载不均**：某些核 100% 利用，某些核空闲
- **缓存亲和性**：任务迁移导致缓存失效，性能下降

本文将为自制 OS 设计一个**完整的多核调度框架**，  
系统性地解决这三大核心问题。

---

## 第一章：调度器的核心问题域

### 1.1 三大核心问题

#### 任务选择（Task Selection）
- **目标**：在单个 CPU 上选择最优任务
- **策略**：时间片轮转、优先级调度、公平调度
- **挑战**：平衡公平性、响应性、吞吐量

#### CPU 选择（CPU Selection）
- **目标**：决定新任务在哪个 CPU 上运行
- **策略**：亲和性绑定、负载感知分配
- **挑战**：减少跨核迁移，保持缓存亲和性

#### 负载均衡（Load Balancing）
- **目标**：确保所有 CPU 负载均衡
- **策略**：定期检查、忙核迁移、空闲核唤醒
- **挑战**：迁移开销 vs 负载收益

### 1.2 调度时机与触发条件

| 触发条件 | 任务选择 | CPU 选择 | 负载均衡 |
|----------|----------|----------|----------|
| **新任务创建** | - | ✓ | - |
| **任务唤醒** | ✓ | ✓ | - |
| **时钟中断** | ✓ | - | ✓ |
| **CPU 空闲** | ✓ | - | ✓ |
| **系统调用** | ✓ | - | - |

> ✅ **完整的调度器必须同时处理这三个维度**！

---

## 第二章：多核调度框架设计

### 2.1 调度域（Scheduling Domain）概念

#### 调度域层次：
- **MC **(Multi-Core)：同一物理 CPU 的核心
- **SMT **(Simultaneous Multi-Threading)：超线程（逻辑核心）

#### 自制 OS 简化：**两级调度域**
```
System
├── Package 0 (物理 CPU 0)
│   ├── CPU 0
│   └── CPU 1
└── Package 1 (物理 CPU 1)
    ├── CPU 2
    └── CPU 3
```

### 2.2 数据结构设计

#### CPU 运行队列：
```c
// sched.h
struct rq {
    raw_spinlock_t lock;
    struct list_head queue;        // 可运行任务队列
    unsigned long nr_running;      // 可运行任务数
    struct task_struct *curr;      // 当前运行任务
    unsigned long cpu;             // CPU 编号
    unsigned long idle_stamp;      // 空闲时间戳
    unsigned long avg_load;        // 平均负载
};
```

#### 调度域：
```c
struct sched_domain {
    struct sched_domain *parent;   // 父域
    struct sched_domain *child;    // 子域
    struct cpumask span;           // 覆盖的 CPU 位图
    unsigned int level;            // 域层级
    unsigned long min_interval;    // 负载均衡最小间隔
    unsigned long max_interval;    // 负载均衡最大间隔
};
```

#### 全局调度状态：
```c
// sched.c
static DEFINE_PER_CPU(struct rq, cpu_rq);
static cpumask_t sched_rq_pending; // 有任务等待的 CPU

// 每个 CPU 的调度域
static DEFINE_PER_CPU(struct sched_domain *, cpu_sched_domains);
```

### 2.3 任务结构扩展

```c
struct task_struct {
    // ... 基础字段
    int on_rq;                     // 0=不在队列, 1=在队列
    int cpu;                       // 当前分配的 CPU
    int wake_cpu;                  // 唤醒时的目标 CPU
    cpumask_t cpus_allowed;        // 允许运行的 CPU 位图
    
    // 调度统计
    u64 sum_exec_runtime;          // 累计执行时间
    u64 prev_sum_exec_runtime;     // 上次调度时的执行时间
    u64 last_ran;                  // 上次运行时间戳
    
    // 负载均衡相关
    struct sched_entity se;        // 调度实体（CFS 用）
    struct list_head rt_run_list;  // 实时任务队列
};
```

---

## 第三章：任务选择机制

### 3.1 运行队列管理

#### 入队操作：
```c
// sched.c
static void enqueue_task(struct task_struct *p, struct rq *rq) {
    if (p->on_rq) return;
    
    // 1. 标记任务在队列中
    p->on_rq = 1;
    p->cpu = rq->cpu;
    
    // 2. 加入运行队列
    list_add_tail(&p->run_list, &rq->queue);
    rq->nr_running++;
    
    // 3. 更新负载统计
    update_rq_load(rq);
    
    // 4. 标记 CPU 有可运行任务
    cpumask_set_cpu(rq->cpu, &sched_rq_pending);
}
```

#### 出队操作：
```c
static void dequeue_task(struct task_struct *p, struct rq *rq) {
    if (!p->on_rq) return;
    
    // 1. 从队列移除
    list_del(&p->run_list);
    rq->nr_running--;
    p->on_rq = 0;
    
    // 2. 更新负载统计
    update_rq_load(rq);
    
    // 3. 清除 CPU 标记（如果队列为空）
    if (rq->nr_running == 0) {
        cpumask_clear_cpu(rq->cpu, &sched_rq_pending);
    }
}
```

### 3.2 调度主循环

#### 核心函数：`__schedule()`
```c
// sched.c
static void __schedule(bool preempt) {
    struct task_struct *prev, *next;
    struct rq *rq;
    int cpu;
    
    cpu = smp_processor_id();
    rq = cpu_rq(cpu);
    prev = rq->curr;
    
    // 1. 获取运行队列锁
    raw_spin_lock(&rq->lock);
    
    // 2. 选择下一个任务
    next = pick_next_task(rq);
    
    // 3. 清除当前任务
    clear_preempt_need_resched();
    
    // 4. 如果任务切换
    if (likely(prev != next)) {
        rq->curr = next;
        prev->on_rq = 0;
        
        // 5. 执行上下文切换
        context_switch(rq, prev, next);
    }
    
    raw_spin_unlock(&rq->lock);
}
```

#### 任务选择策略：
```c
// sched/rt.c
static struct task_struct *pick_next_task_rt(struct rq *rq) {
    // 实时任务：选择最高优先级
    if (!list_empty(&rq->rt_queue)) {
        return list_first_entry(&rq->rt_queue, struct task_struct, rt_run_list);
    }
    return NULL;
}

// sched/fair.c
static struct task_struct *pick_next_task_fair(struct rq *rq) {
    // 公平调度：选择 vruntime 最小的任务
    struct sched_entity *se = __pick_first_entity(&rq->cfs);
    return se ? task_of(se) : NULL;
}

// 调度器入口
static struct task_struct *pick_next_task(struct rq *rq) {
    // 1. 优先实时任务
    struct task_struct *p = pick_next_task_rt(rq);
    if (p) return p;
    
    // 2. 回退到公平调度
    return pick_next_task_fair(rq);
}
```

---

## 第四章：CPU 选择与亲和性

### 4.1 CPU 选择策略

#### 新任务 CPU 分配：
```c
// sched/cpupri.c
static int select_task_rq(struct task_struct *p, int prev_cpu, int sd_flags) {
    // 1. 检查亲和性
    if (!cpumask_test_cpu(prev_cpu, &p->cpus_allowed)) {
        prev_cpu = cpumask_any(&p->cpus_allowed);
    }
    
    // 2. 检查缓存亲和性
    if (cpu_rq(prev_cpu)->nr_running < 2) {
        return prev_cpu; // 原 CPU 负载低，保持亲和性
    }
    
    // 3. 选择最空闲的 CPU
    return find_least_loaded_cpu(&p->cpus_allowed);
}
```

#### 任务唤醒 CPU 选择：
```c
// sched/wake_q.c
static int wake_wakee_cpu(struct task_struct *p, struct task_struct *curr) {
    int target = p->wake_cpu;
    
    // 1. 如果唤醒 CPU 空闲，直接选择
    if (idle_cpu(target)) {
        return target;
    }
    
    // 2. 检查当前 CPU 是否更优
    if (cpu_rq(smp_processor_id())->nr_running < 
        cpu_rq(target)->nr_running) {
        return smp_processor_id();
    }
    
    return target;
}
```

### 4.2 CPU 亲和性管理

#### 设置亲和性：
```c
// sys_sched.c
long sys_sched_setaffinity(pid_t pid, unsigned int len, unsigned long *user_mask_ptr) {
    struct task_struct *p;
    cpumask_var_t new_mask;
    
    // 1. 复制用户态 CPU 位图
    if (copy_from_user(new_mask, user_mask_ptr, len)) {
        return -EFAULT;
    }
    
    // 2. 找到目标任务
    p = find_task_by_vpid(pid);
    if (!p) return -ESRCH;
    
    // 3. 更新亲和性
    p->cpus_allowed = *new_mask;
    
    // 4. 如果当前 CPU 不在允许列表中，迁移任务
    if (!cpumask_test_cpu(task_cpu(p), &p->cpus_allowed)) {
        migrate_task_to(p, cpumask_any(&p->cpus_allowed));
    }
    
    return 0;
}
```

---

## 第五章：负载均衡机制

### 5.1 负载均衡触发

#### 定时均衡：
```c
// sched/fair.c
static void run_rebalance_domains(struct irq_work *work) {
    int this_cpu = smp_processor_id();
    struct rq *rq = cpu_rq(this_cpu);
    struct sched_domain *sd;
    
    // 遍历所有调度域
    for_each_domain(this_cpu, sd) {
        if (time_after(jiffies, sd->last_balance + sd->balance_interval)) {
            load_balance(this_cpu, rq, sd, CPU_IDLE);
            sd->last_balance = jiffies;
        }
    }
}

// 时钟中断中触发
void scheduler_tick(void) {
    // ... 其他逻辑
    
    // 检查是否需要负载均衡
    if (unlikely(rebalance_domains_need()))
        irq_work_queue(&rebalance_irq_work);
}
```

#### 空闲均衡：
```c
// sched/idle.c
void sched_idle_balance(void) {
    struct rq *rq = this_rq();
    
    // 如果有可运行任务，无需均衡
    if (rq->nr_running) return;
    
    // 从其他 CPU 拉取任务
    load_balance(this_cpu, rq, sd, CPU_IDLE);
}
```

### 5.2 负载均衡算法

#### 负载计算：
```c
// sched/loadavg.c
static unsigned long source_load(int cpu, int type) {
    struct rq *rq = cpu_rq(cpu);
    unsigned long total = rq->nr_running;
    
    // 加权负载：运行任务数 + 睡眠任务影响
    return total;
}

static unsigned long target_load(int cpu, int type) {
    return source_load(cpu, type);
}
```

#### 任务迁移：
```c
// sched/migrate.c
static int move_one_task(struct rq *this_rq, int this_cpu, 
                        struct rq *busiest, struct sched_domain *sd) {
    struct list_head *tasks = &busiest->queue;
    struct task_struct *p, *n;
    
    // 遍历最忙 CPU 的任务队列
    list_for_each_entry_safe(p, n, tasks, run_list) {
        // 检查亲和性
        if (!cpumask_test_cpu(this_cpu, &p->cpus_allowed))
            continue;
            
        // 迁移任务
        deactivate_task(busiest, p, 0);
        set_task_cpu(p, this_cpu);
        activate_task(this_rq, p, 0);
        
        // 更新负载统计
        update_rq_load(this_rq);
        update_rq_load(busiest);
        
        return 1;
    }
    return 0;
}
```

### 5.3 负载均衡策略

#### 域级均衡策略：
| 调度域 | 均衡策略 | 间隔 |
|--------|----------|------|
| **SMT **(超线程) | 轻量均衡（仅迁移休眠任务） | 1ms |
| **MC **(多核) | 中等均衡（迁移可运行任务） | 10ms |
| **NUMA** | 重量均衡（跨节点迁移） | 100ms |

#### 迁移开销评估：
- **缓存失效成本**：迁移任务导致 L1/L2 缓存失效
- **TLB 刷新成本**：跨核迁移需刷新 TLB
- **同步开销**：获取多个运行队列锁

> ✅ **只有当负载收益 > 迁移开销时，才执行迁移**！

---

## 第六章：调度器初始化与启动

### 6.1 多核调度器初始化

```c
// sched.c
void __init sched_init(void) {
    int cpu;
    
    // 1. 初始化每个 CPU 的运行队列
    for_each_possible_cpu(cpu) {
        struct rq *rq = cpu_rq(cpu);
        
        raw_spin_lock_init(&rq->lock);
        INIT_LIST_HEAD(&rq->queue);
        rq->nr_running = 0;
        rq->curr = &idle_task;
        rq->cpu = cpu;
        rq->idle_stamp = jiffies;
        rq->avg_load = 0;
    }
    
    // 2. 初始化调度域
    sched_init_domains();
    
    // 3. 初始化负载均衡工作队列
    init_irq_work(&rebalance_irq_work, run_rebalance_domains);
}
```

### 6.2 调度域构建

```c
// sched/topology.c
static void sched_init_domains(void) {
    int cpu;
    
    for_each_possible_cpu(cpu) {
        struct sched_domain *sd, *p;
        
        // 1. 构建 MC 域（多核）
        sd = build_sched_domain(&cpu_coregroup_mask, SD_INIT_MC);
        cpu_rq(cpu)->sd = sd;
        
        // 2. 构建 NUMA 域（如果存在）
        if (numa_aware()) {
            p = build_sched_domain(&numa_nodes_mask, SD_INIT_NUMA);
            sd->parent = p;
            p->child = sd;
        }
    }
}
```

### 6.3 Idle 任务初始化

```c
// sched/idle.c
void __cpuinit init_idle(struct task_struct *idle, int cpu) {
    struct rq *rq = cpu_rq(cpu);
    
    // 1. 设置 idle 任务
    idle->on_rq = 0;
    idle->cpu = cpu;
    rq->curr = idle;
    
    // 2. 设置亲和性（仅允许在本 CPU 运行）
    cpumask_clear(&idle->cpus_allowed);
    cpumask_set_cpu(cpu, &idle->cpus_allowed);
}
```

---

## 第七章：性能优化与调试

### 7.1 关键性能指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **上下文切换次数** | 每秒切换次数 | < 1000 |
| **负载均衡频率** | 每秒均衡次数 | < 10 |
| **任务迁移次数** | 每秒迁移次数 | < 100 |
| **CPU 利用率方差** | 各核利用率差异 | < 10% |

### 7.2 调试接口

#### `/proc/sched_debug`：
```c
// sched/debug.c
static int sched_debug_show(struct seq_file *m, void *v) {
    int cpu;
    
    for_each_online_cpu(cpu) {
        struct rq *rq = cpu_rq(cpu);
        seq_printf(m, "CPU%d:\n", cpu);
        seq_printf(m, "  nr_running: %lu\n", rq->nr_running);
        seq_printf(m, "  avg_load: %lu\n", rq->avg_load);
        seq_printf(m, "  curr: %s\n", rq->curr->comm);
    }
    return 0;
}
```

#### 调度统计：
```c
// sched/stats.c
void update_sched_stats(struct task_struct *p) {
    p->sched_info.run_delay += 
        (jiffies - p->last_ran) * jiffies_to_ns(1);
    p->sched_info.pcount++;
}
```

### 7.3 优化方向

#### 锁优化：
- **无锁运行队列**：使用 RCU 保护只读操作
- **分层锁**：调度域锁 + CPU 锁

#### 亲和性优化：
- **NUMA 亲和性**：优先分配本地内存
- **缓存行对齐**：运行队列结构体对齐

#### 负载均衡优化：
- **预测性均衡**：基于历史负载预测
- **增量均衡**：每次只迁移必要任务

---

## 结论：构建完整的多核调度框架

现代调度器是操作系统**最复杂的子系统之一**。  
通过系统性的设计，  
我们解决了：
- **任务选择**：通过可扩展的调度类
- **CPU 选择**：通过亲和性与负载感知
- **负载均衡**：通过分层调度域

此框架为后续实现 **CFS、实时调度、节能调度** 奠定了坚实基础。  
真正的调度器，始于对多核并行本质的深刻理解。

---

## 附录：关键数据结构与接口速查

### 核心数据结构
| 结构 | 作用 |
|------|------|
| `struct rq` | CPU 运行队列 |
| `struct sched_domain` | 调度域 |
| `struct task_struct` | 任务描述符 |

### 调度接口
```c
void schedule(void);                          // 主调度函数
void enqueue_task(struct task_struct *p, struct rq *rq);
void dequeue_task(struct task_struct *p, struct rq *rq);
int select_task_rq(struct task_struct *p, int prev_cpu, int sd_flags);
void load_balance(int this_cpu, struct rq *this_rq, 
                  struct sched_domain *sd, enum cpu_idle_type idle);
```

### 调试接口
```c
/proc/sched_debug    # 调度器状态
/proc/schedstat      # 调度统计
```

> **注**：本文所有代码均为简化实现，实际使用需添加错误处理、边界检查、安全审计等。