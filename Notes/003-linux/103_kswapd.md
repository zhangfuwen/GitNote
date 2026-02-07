
在当前 Linux 内核（截至 2025 年主流版本，如 6.1–6.9）中，`kswapd`（即通常所说的 “swapd”）是负责**异步内存回收**的核心内核线程。它并不直接执行“swap out”，而是负责在后台回收内存页（包括匿名页换出到 swap 分区，以及文件缓存页释放），以维持系统的内存水位。

---

## ✅ 一、kswapd 的核心职责

- 监控内存域（zone）的空闲内存水平。
- 当空闲内存低于 **low watermark（低水位线）** 时，唤醒 `kswapd`。
- `kswapd` 持续回收页面，直到内存达到 **high watermark（高水位线）**。
- 回收对象包括：
  - 匿名页（Anonymous pages）→ 可能写入 **swap 分区**
  - 文件页缓存（Page Cache）→ 可直接释放或写回磁盘

> ⚠️ 注意：`kswapd` 不处理紧急情况。当内存极度紧张时，由直接回收（direct reclaim）机制处理，不依赖 `kswapd`。

---

## ✅ 二、kswapd 使用的核心算法与机制

### 1. **LRU（Least Recently Used）双向链表机制**

Linux 使用 **分层 LRU 链表** 来管理页面冷热状态：

#### 🔗 主要链表（每个内存域 zone 维护）：
| 链表 | 含义 |
|------|------|
| `inactive_anon` | 不活跃的匿名页（可能被 swap out） |
| `active_anon`   | 活跃的匿名页（暂时不换出） |
| `inactive_file` | 不活跃的文件缓存页（可释放） |
| `active_file`   | 活跃的文件缓存页（仍在使用） |

> 🔄 页面在 `active` 和 `inactive` 之间迁移，基于访问频率和时间判断“冷热”。

#### 🔍 页面晋升/降级规则：
- 如果一个页面被访问（通过页表项的 accessed bit 检测），则从 `inactive` → `active`
- 定期扫描，将长时间未访问的 `active` 页面移入 `inactive`
- 只有 `inactive` 列表中的页面才会被 `kswapd` 考虑回收

📌 这不是精确 LRU，而是 **近似 LRU（Pseudo-LRU）**，因为维护完整 LRU 开销太大。

---

### 2. **Working Set Detection（工作集检测）**

内核通过周期性扫描（由 `kswapd` 或其他回收路径触发）来识别进程的工作集（正在使用的内存集合）：

- 扫描 `active` 列表中的页面，检查是否最近被访问过
- 若未访问，则降级到 `inactive`
- 若频繁访问，则保留在 `active`

目的：避免把“正在使用”的页面错误地换出。

---

### 3. **Swap Token（已废弃）**

早期 Linux 曾使用 **Swap Token** 算法来防止多个进程同时发生大量缺页时的“抖动”（thrashing）：

- 给“最需要 swap 的进程”一个 token，允许其优先保留内存
- 其他进程更积极地释放内存

❌ **现状**：自 Linux 4.0+ 起已被移除，因效果有限且增加复杂度。

---

### 4. **Page Selection Algorithm：Shuffle-based Pageout**

现代 Linux（尤其是 5.10+）引入了改进的页面选择策略：

- 在扫描 `inactive` 列表时，并非顺序处理，而是采用**随机化或轮询方式**，避免某些页面长期得不到回收机会。
- 结合 **PFRA（Page Frame Reclaim Algorithm）** 改进公平性。

---

### 5. **Writeback 与 IO 调度协同**

当页面需要写回磁盘（如 dirty file cache 或匿名页写入 swap），`kswapd` 会：
- 将脏页提交给 **writeback 子系统**
- 控制写回速率，避免 I/O 风暴
- 使用 **balance_dirty_pages()** 机制进行节流

这不是页面置换算法，但影响整体 swap 性能。

---

### 6. **NUMA 支持：per-node kswapd**

在 NUMA 系统中，每个节点有自己的 `kswapdN`（如 `kswapd0`, `kswapd1`），分别管理本地内存节点的回收。

- 减少跨节点内存访问开销
- 根据 local vs remote 访问延迟调整回收优先级

---

### 7. **Memory Pressure Prediction（轻量预测）**

虽然没有 AI，但现代内核有一些**启发式预测机制**：

- 根据过去一段时间的 page fault 频率，预判未来压力
- 动态调整 `kswapd` 唤醒频率（如使用 `zone->nr_jobs` 计数器）
- 使用 **proportional scan**：根据页面类型（file/anon）的比例决定扫描权重

---

## ✅ 三、kswapd 不使用的算法（常见误解）

| 算法 | 是否使用 | 说明 |
|------|----------|------|
| FIFO | ❌ | 太简单，无法反映实际使用模式 |
| Clock / Second Chance | ❌ | 类似 LRU 的近似实现，但 Linux 用的是 list-based LRU |
| OPT（最优置换） | ❌ | 不现实，需预知未来 |
| LFU（最不经常使用） | ❌ | 内核无频率计数器支持 |
| AI/ML 模型 | ❌ | 目前仍为研究阶段，未集成进主线内核 |

---

## ✅ 四、总结：当前 kswapd 的核心算法体系

| 层级          | 算法/机制                            | 作用        | 备注                                                          |
| ----------- | -------------------------------- | --------- | ----------------------------------------------------------- |
| **页面组织**    | Active/Inactive LRU 链表           | 区分冷热页面    | 冷热的记录是由硬件完成的，是记录在页表标志位的，无法完整记录页面访问的历史（因为每个load/store指令都要记录） |
| **冷热判定**    | Accessed bit 扫描 + 工作集检测          | 决定是否降级页面  |                                                             |
| **回收顺序**    | 优先回收 `inactive` 中的 file 和 anon 页 | 最小化性能影响   |                                                             |
| **I/O 控制**  | Writeback 节流 + 异步提交              | 防止 I/O 抖动 |                                                             |
| **NUMA 优化** | per-node kswapd                  | 提升局部性     |                                                             |
| **阈值控制**    | Watermark（min/low/high）          | 触发与停止回收   |                                                             |

---

## ✅ 五、未来方向（研究中）

尽管目前 `kswapd` 仍基于传统机制，但以下方向正在探索中：

- **基于机器学习的页面热度预测**（如 Google 的 DQN 页置换）
- **eBPF + 用户空间控制器**：让用户程序干预 swap 决策
- **CXL/异构内存管理**：对不同层级内存（DRAM, PMEM, SWAP）做分级调度

---

### 📌 结论

> 当前 Linux 的 `kswapd` 主要依赖 **Active/Inactive LRU 链表机制 + 工作集检测 + 水位线控制** 来决定何时回收内存、回收哪些页面。  
> 它不是单一算法，而是一套**综合的内存压力响应系统**，重点在于平衡性能、公平性和 I/O 效率。

它没有使用复杂的 AI 或经典替换算法（如 Clock、LFU），而是采用**高效、可扩展的近似 LRU 框架**，适合大规模生产环境。