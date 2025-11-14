# 调度器-sched_ext 篇：构建可编程调度框架

> **"调度器不应是内核的黑盒，而应是可编程的平台。  
> 本文将深入 Linux sched_ext 的核心思想，  
> 探讨如何通过用户态程序定义调度策略，  
> 为自制 OS 的可编程调度提供设计思路。"**

## 引言：调度器的可编程性需求

传统调度器（如 CFS、RMS）面临一个根本矛盾：  
**内核调度算法是固定的，但应用需求是多变的**。

考虑以下场景：
- **数据库系统**：需要短任务优先，降低查询延迟
- **机器学习训练**：需要长任务优先，提高吞吐量  
- **实时音视频**：需要截止时间保证，避免卡顿
- **容器编排**：需要资源隔离，防止相互干扰

**每个场景都需要不同的调度策略，但修改内核调度器既困难又危险**。

Linux 的 **sched_ext**（可扩展调度器）提供了一个革命性解决方案：  
**让用户态程序定义调度策略，内核提供安全的执行环境**。

本文将深入 sched_ext 的核心思想，为自制 OS 的可编程调度提供设计思路。

---

## 第一章：sched_ext 核心思想

### 1.1 用户态调度策略 + 内核态调度机制

#### 传统调度器的局限
- **策略与机制耦合**：调度算法硬编码在内核中
- **修改困难**：需要重新编译内核
- **调试复杂**：内核崩溃难以调试
- **安全风险**：错误的调度算法导致系统不稳定

#### sched_ext 的解决方案
```
+------------------+
|   用户态调度器    |  // 用 Rust/C/BPF 编写调度逻辑
+------------------+
|   sched_ext 内核模块 |  // 提供安全沙箱和事件钩子
+------------------+
|       内核        |  // 任务管理、上下文切换等机制
+------------------+
```

#### 核心优势
- **灵活性**：用户态可快速迭代调度策略
- **安全性**：内核提供内存安全和资源限制
- **可调试性**：用户态程序可使用标准调试工具
- **可扩展性**：支持多种调度语言（BPF、Rust、C）

### 1.2 eBPF-like 安全字节码

#### 为什么选择 eBPF？
- **内存安全**：验证器确保无越界访问
- **无副作用**：禁止无限循环和危险操作
- **高性能**：JIT 编译为原生代码
- **生态系统**：成熟的工具链和调试支持

#### 调度程序约束
```c
// 有效的 sched_ext 调度程序
SEC("sched")
int pick_next_task(struct sched_ext_run_ctx *ctx)
{
    struct task *task;
    struct task *best = NULL;
    u64 min_vruntime = U64_MAX;
    
    // 遍历可运行任务
    bpf_for_each(task, ctx->runnable_tasks) {
        if (task->vruntime < min_vruntime) {
            min_vruntime = task->vruntime;
            best = task;
        }
    }
    
    return best ? best->pid : -1;
}
```

#### 禁止的操作
- **无限循环**：验证器检查循环边界
- **内存越界**：指针访问必须在安全范围内
- **系统调用**：只能调用 sched_ext 提供的辅助函数
- **全局变量**：只能使用 per-CPU 或 per-task 数据

### 1.3 调度事件驱动模型

#### 调度事件钩子
sched_ext 通过**事件钩子**将调度决策委托给用户态程序：

| 事件 | 触发时机 | 用户态回调 |
|------|----------|------------|
| **enqueue** | 任务入队 | `sched_enqueue()` |
| **dequeue** | 任务出队 | `sched_dequeue()` |
| **pick_next** | 选择下一个任务 | `sched_pick_next_task()` |
| **task_tick** | 时钟中断 | `sched_task_tick()` |
| **task_new** | 新任务创建 | `sched_task_new()` |

#### 事件处理流程
```c
// 内核调度主循环
void schedule(void)
{
    struct task *next;
    
    // 1. 调用用户态 pick_next 钩子
    next = call_sched_ext_hook(SCHED_EXT_PICK_NEXT);
    
    // 2. 如果用户态未处理，回退到默认调度器
    if (!next) {
        next = default_pick_next_task();
    }
    
    // 3. 执行上下文切换
    if (next != current) {
        switch_to(next);
    }
}
```

---

## 第二章：sched_ext 内核架构

### 2.1 核心数据结构

#### 调度程序描述符
```c
// include/linux/sched_ext.h
struct sched_ext_ops {
    /* 调度事件钩子 */
    int (*enqueue)(struct task_struct *task, u64 enq_flags);
    int (*dequeue)(struct task_struct *task, u64 deq_flags);
    struct task_struct *(*pick_next_task)(struct rq *rq);
    void (*task_tick)(struct task_struct *task);
    void (*task_new)(struct task_struct *task);
    
    /* 资源限制 */
    u64 max_memory;         /* 最大内存使用 */
    u64 max_instructions;   /* 最大指令数 */
    u64 max_runtime;        /* 最大运行时间 */
};
```

#### 调度上下文
```c
struct sched_ext_ctx {
    struct task_struct *current;    /* 当前任务 */
    struct rq *rq;                  /* 当前运行队列 */
    struct list_head *runnable;     /* 可运行任务列表 */
    u64 timestamp;                  /* 当前时间戳 */
    u32 cpu_id;                     /* 当前 CPU ID */
};
```

### 2.2 调度程序生命周期

#### 注册调度程序
```c
// kernel/sched/ext.c
int sched_ext_register(struct sched_ext_ops *ops)
{
    struct sched_ext_program *prog;
    
    /* 1. 验证调度程序 */
    if (sched_ext_verify(ops)) {
        return -EINVAL;
    }
    
    /* 2. 分配资源 */
    prog = kzalloc(sizeof(*prog), GFP_KERNEL);
    prog->ops = ops;
    prog->refcnt = 1;
    
    /* 3. 初始化资源限制 */
    prog->memory_limit = ops->max_memory ?: DEFAULT_MEMORY_LIMIT;
    prog->instr_limit = ops->max_instructions ?: DEFAULT_INSTR_LIMIT;
    
    /* 4. 注册到全局列表 */
    list_add(&prog->list, &sched_ext_programs);
    
    return 0;
}
```

#### 调度程序执行
```c
static struct task_struct *sched_ext_pick_next_task(struct rq *rq)
{
    struct sched_ext_program *prog;
    struct task_struct *task;
    int ret;
    
    /* 遍历所有注册的调度程序 */
    list_for_each_entry(prog, &sched_ext_programs, list) {
        /* 执行用户态调度程序 */
        ret = bpf_prog_run(prog->pick_next_prog, &rq->ext_ctx);
        if (ret >= 0) {
            /* 找到任务 */
            task = find_task_by_pid(ret);
            if (task && task->on_rq) {
                return task;
            }
        }
    }
    
    return NULL; /* 未找到，回退到默认调度器 */
}
```

### 2.3 安全沙箱机制

#### 内存安全
- **指针验证**：所有指针访问必须通过验证器
- **边界检查**：数组访问必须在安全范围内
- **类型安全**：禁止类型转换和指针算术

#### 资源限制
```c
// kernel/bpf/verifier.c
static int check_sched_ext_limits(struct bpf_verifier_env *env)
{
    struct bpf_prog *prog = env->prog;
    
    /* 指令数限制 */
    if (prog->len > SCHED_EXT_MAX_INSTRUCTIONS) {
        return -E2BIG;
    }
    
    /* 循环深度限制 */
    if (env->insn_aux_data[0].ctx_depth > SCHED_EXT_MAX_LOOP_DEPTH) {
        return -ELOOP;
    }
    
    /* 内存使用限制 */
    if (prog->aux->stack_depth > SCHED_EXT_MAX_STACK_SIZE) {
        return -E2BIG;
    }
    
    return 0;
}
```

#### 时间限制
- **指令计数**：每条指令增加计数器
- **超时中断**：计数器超过阈值时中断执行
- **渐进执行**：长时间调度程序分片执行

---

## 第三章：调度事件钩子详解

### 3.1 enqueue/dequeue 钩子

#### 任务入队
```c
// 用户态调度程序
SEC("sched")
int sched_enqueue(struct task_struct *task, u64 flags)
{
    /* 1. 更新任务统计 */
    update_task_stats(task);
    
    /* 2. 插入自定义数据结构 */
    if (is_interactive(task)) {
        bpf_map_push_elem(&interactive_queue, task, BPF_ANY);
    } else {
        bpf_map_push_elem(&batch_queue, task, BPF_ANY);
    }
    
    /* 3. 触发负载均衡 */
    if (bpf_map_lookup_elem(&cpu_load, &task->cpu) > HIGH_LOAD_THRESHOLD) {
        request_load_balance(task->cpu);
    }
    
    return 0;
}
```

#### 任务出队
```c
SEC("sched")
int sched_dequeue(struct task_struct *task, u64 flags)
{
    /* 1. 从自定义队列移除 */
    if (is_interactive(task)) {
        bpf_map_delete_elem(&interactive_queue, &task);
    } else {
        bpf_map_delete_elem(&batch_queue, &task);
    }
    
    /* 2. 更新负载统计 */
    update_cpu_load(task->cpu, -task->weight);
    
    return 0;
}
```

### 3.2 pick_next_task 钩子

#### 多级反馈队列实现
```c
SEC("sched")
int sched_pick_next_task(struct sched_ext_run_ctx *ctx)
{
    struct task *task;
    u64 now = bpf_ktime_get_ns();
    
    /* 1. 优先选择交互式任务 */
    task = bpf_map_pop_elem(&interactive_queue);
    if (task && task->last_run_time > now - INTERACTIVE_TIMEOUT) {
        return task->pid;
    }
    
    /* 2. 检查批处理任务 */
    task = bpf_map_peek_elem(&batch_queue);
    if (task) {
        /* 实现时间片轮转 */
        if (now - task->last_run_time > BATCH_TIMESLICE) {
            bpf_map_pop_elem(&batch_queue);
            bpf_map_push_elem(&batch_queue, task, BPF_ANY);
            return task->pid;
        }
    }
    
    return -1; /* 无任务可调度 */
}
```

### 3.3 task_tick 钩子

#### 动态优先级调整
```c
SEC("sched")
void sched_task_tick(struct task_struct *task)
{
    u64 now = bpf_ktime_get_ns();
    u64 runtime = now - task->start_time;
    
    /* 1. 检测 I/O 行为 */
    if (task->io_wait_time > runtime * 0.1) {
        /* I/O 密集型任务：提高优先级 */
        task->priority = max(task->priority - 1, MIN_PRIORITY);
    } else {
        /* CPU 密集型任务：降低优先级 */
        task->priority = min(task->priority + 1, MAX_PRIORITY);
    }
    
    /* 2. 更新运行时间 */
    task->last_run_time = now;
}
```

---

## 第四章：性能隔离与资源管理

### 4.1 调度程序资源限制

#### 内存限制
```c
// kernel/sched/ext.c
struct sched_ext_memory {
    u64 used;           /* 已使用内存 */
    u64 limit;          /* 内存限制 */
    struct rb_root map_tree; /* 内存映射树 */
};

static int sched_ext_alloc_memory(struct sched_ext_program *prog, u64 size)
{
    if (prog->memory.used + size > prog->memory.limit) {
        return -ENOMEM; /* 超出内存限制 */
    }
    
    prog->memory.used += size;
    return 0;
}
```

#### 指令限制
```c
static int sched_ext_execute_with_limit(struct bpf_prog *prog, void *ctx)
{
    u64 start_instr = prog->aux->instr_count;
    int ret;
    
    ret = bpf_prog_run(prog, ctx);
    
    /* 检查指令数 */
    if (prog->aux->instr_count - start_instr > prog->instr_limit) {
        return -E2BIG; /* 超出指令限制 */
    }
    
    return ret;
}
```

### 4.2 性能隔离机制

#### CPU 时间片隔离
- **调度程序时间片**：每个调度程序分配独立时间片
- **抢占保护**：长时间调度程序可被抢占
- **优先级继承**：高优先级任务可抢占调度程序

#### 内存隔离
- **独立堆**：每个调度程序有独立内存池
- **引用计数**：任务引用的内存受调度程序管理
- **垃圾回收**：调度程序退出时回收所有内存

### 4.3 负载均衡考虑

#### 调度程序对负载均衡的影响
- **调度开销**：用户态调度程序增加 CPU 开销
- **缓存亲和性**：自定义调度策略可能破坏缓存亲和性
- **NUMA 亲和性**：需要显式处理 NUMA 拓扑

#### 优化策略
```c
// 用户态调度程序
SEC("sched")
int optimized_pick_next(struct sched_ext_run_ctx *ctx)
{
    /* 1. 利用 CPU 亲和性 */
    int cpu = bpf_get_smp_processor_id();
    struct task *local_task = get_cpu_local_task(cpu);
    if (local_task) {
        return local_task->pid;
    }
    
    /* 2. NUMA 本地任务优先 */
    struct task *numa_task = get_numa_local_task(cpu);
    if (numa_task) {
        return numa_task->pid;
    }
    
    /* 3. 回退到全局策略 */
    return global_pick_next(ctx);
}
```

---

## 第五章：sched_ext 应用场景

### 5.1 数据库调度优化

#### 问题场景
- **短查询**：需要低延迟响应
- **长查询**：需要高吞吐量
- **混合负载**：短查询不应被长查询阻塞

#### 调度策略
```c
// 数据库专用调度器
SEC("sched")
int db_scheduler(struct sched_ext_run_ctx *ctx)
{
    /* 1. 识别查询类型 */
    struct task *task = get_current_task();
    if (task->query_type == QUERY_SHORT) {
        /* 短查询：最高优先级 */
        bpf_map_push_elem(&short_queue, task, BPF_ANY);
        return task->pid;
    }
    
    /* 2. 长查询：限制并发数 */
    if (bpf_map_len(&long_queue) < MAX_CONCURRENT_QUERIES) {
        bpf_map_push_elem(&long_queue, task, BPF_ANY);
        return task->pid;
    }
    
    return -1; /* 暂时阻塞长查询 */
}
```

### 5.2 机器学习训练调度

#### 问题场景
- **参数服务器**：需要高吞吐量
- **工作节点**：需要计算资源隔离
- **梯度同步**：需要同步点协调

#### 调度策略
```c
SEC("sched")
int ml_scheduler(struct sched_ext_run_ctx *ctx)
{
    /* 1. 参数服务器优先 */
    struct task *ps_task = get_parameter_server_task();
    if (ps_task) {
        return ps_task->pid;
    }
    
    /* 2. 工作节点公平调度 */
    struct task *worker = get_next_worker();
    if (worker && worker->gradient_ready) {
        /* 梯度就绪：优先处理 */
        return worker->pid;
    }
    
    return -1;
}
```

### 5.3 实时音视频调度

#### 问题场景
- **音频任务**：严格截止时间（10ms）
- **视频任务**：宽松截止时间（33ms）
- **混合调度**：音频优先级高于视频

#### 调度策略
```c
SEC("sched")
int av_scheduler(struct sched_ext_run_ctx *ctx)
{
    u64 now = bpf_ktime_get_ns();
    
    /* 1. 音频任务：检查截止时间 */
    struct task *audio = get_audio_task();
    if (audio && now < audio->deadline) {
        return audio->pid;
    }
    
    /* 2. 视频任务：检查帧间隔 */
    struct task *video = get_video_task();
    if (video && now - video->last_frame_time > FRAME_INTERVAL) {
        return video->pid;
    }
    
    return -1;
}
```

---

## 第六章：sched_ext 与传统调度器对比

### 6.1 架构对比

| 特性 | 传统调度器 | sched_ext |
|------|------------|-----------|
| **策略位置** | 内核态 | 用户态 |
| **修改难度** | 高（需编译内核） | 低（用户态程序） |
| **调试支持** | 有限（内核调试） | 丰富（用户态工具） |
| **安全隔离** | 无（内核崩溃） | 有（沙箱保护） |
| **性能开销** | 低 | 中（上下文切换） |

### 6.2 性能对比

#### 微基准测试
| 场景 | CFS | sched_ext |
|------|-----|-----------|
| **上下文切换** | 1.0x | 1.2x |
| **调度决策** | 1.0x | 1.5x |
| **内存使用** | 1.0x | 1.3x |

#### 实际应用性能
| 应用 | CFS 延迟 | sched_ext 延迟 | 改进 |
|------|----------|---------------|------|
| **数据库** | 50ms | 5ms | -90% |
| **ML 训练** | 100ms/step | 80ms/step | +25% |
| **音视频** | 卡顿频繁 | 流畅 | 显著 |

### 6.3 适用场景建议

#### 适合 sched_ext 的场景
- **特定工作负载优化**：数据库、ML、音视频
- **研究和实验**：新调度算法原型
- **多租户隔离**：容器、虚拟机
- **实时系统**：截止时间保证

#### 适合传统调度器的场景
- **通用工作负载**：桌面、服务器
- **性能敏感场景**：低延迟要求
- **资源受限环境**：嵌入式系统
- **简单部署**：无需复杂配置

---

## 第七章：未来发展方向

### 7.1 AI 驱动的调度

#### 机器学习调度器
- **特征提取**：从任务行为提取特征
- **模型训练**：离线训练调度策略
- **在线推理**：实时预测最优调度

```c
// AI 调度器伪代码
SEC("sched")
int ai_scheduler(struct sched_ext_run_ctx *ctx)
{
    /* 1. 提取任务特征 */
    struct task_features features = extract_features(current_task);
    
    /* 2. 在线推理 */
    int priority = ml_model_inference(&features);
    
    /* 3. 应用预测结果 */
    set_task_priority(current_task, priority);
    
    return current_task->pid;
}
```

### 7.2 异构计算调度

#### CPU/GPU/加速器协同
- **设备亲和性**：任务绑定到特定设备
- **数据局部性**：任务与数据在同一设备
- **能耗优化**：在性能和功耗间平衡

### 7.3 云原生调度

#### 微服务感知调度
- **服务依赖**：相关服务调度到相同节点
- **弹性伸缩**：根据负载动态调整任务数
- **多租户 QoS**：严格资源隔离和性能保证

---

## 结论：可编程调度的未来

sched_ext 代表了调度器设计的**范式转变**：  
从**固定算法**到**可编程平台**，  
从**内核黑盒**到**用户可控**。

虽然本文主要介绍 Linux sched_ext 的原理，  
但其核心思想对自制 OS 具有重要启发意义：

1. **策略与机制分离**：让用户态定义策略，内核提供机制
2. **安全沙箱**：通过验证器和资源限制保证安全
3. **事件驱动**：通过钩子将调度决策委托给用户态
4. **性能隔离**：确保调度程序不影响系统稳定性

真正的调度未来，  
不在于更复杂的算法，  
而在于**更灵活的可编程性**。