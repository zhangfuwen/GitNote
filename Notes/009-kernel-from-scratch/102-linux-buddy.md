
# 内存管理-物理篇：Buddy 系统深度剖析 —— 从理论到工业级实现

> **“Buddy 分配器远不止‘分裂与合并’那么简单。  
> 本文将深入 Linux 内核源码级实现，系统解析水位线、迁移类型、每 CPU 页框缓存、复合页等核心机制，  
> 并提供可运行的工业级实现框架。”**

## 引言：Buddy 系统的历史地位与挑战

在操作系统的内存管理子系统中，**物理内存分配器**（Physical Memory Allocator）承担着最基础也最关键的职责：高效管理物理页帧（Page Frame），为内核和用户程序提供内存服务。自 1960 年代 Knuth 提出 Buddy 系统以来，这一算法因其简洁性和对碎片的有效控制，成为几乎所有现代操作系统的标准选择。

Linux 内核的 Buddy 分配器经过 20 多年的演进，已从最初的简单实现发展为包含**水位线管理、迁移类型、每 CPU 缓存、复合页支持、NUMA 感知**等高级特性的复杂系统。这些特性共同解决了实际生产环境中面临的诸多挑战：

- **内存耗尽死锁**：内核关键路径需要应急内存
- **外部碎片**：不可移动页导致大块内存无法分配
- **多核性能瓶颈**：全局锁竞争严重
- **TLB 压力**：4KB 小页导致 TLB 条目快速耗尽
- **NUMA 延迟**：跨节点内存访问性能下降

本文将系统性地剖析 Buddy 系统的各个核心组件，从理论基础到工业实现，从单核到多核，从 4KB 页到 1GB 大页。我们将深入 Linux 内核源码（基于 5.15 版本），解析关键数据结构和算法，并提供可运行的简化实现框架。

---

## 第一章：理论基石与地址算术

### 1.1 Buddy 系统的数学本质

Buddy 系统的核心思想源于**二进制地址的对称性**。对于一个大小为 $2^n$ 个页的内存块，其 Buddy 块的地址可以通过简单的**异或运算**得到：

$$\text{buddy\_addr} = \text{addr} \oplus 2^n \times \text{PAGE\_SIZE}$$

这里的 $2^n \times \text{PAGE\_SIZE}$ 是块的大小（以字节为单位）。在页帧号（PFN）空间中，公式简化为：

$$\text{buddy\_pfn} = \text{pfn} \oplus 2^n$$

#### 证明过程：
假设内存被划分为大小相等的块，每个块包含 $2^n$ 个连续页。将内存按 $2^{n+1}$ 个页为单位分组，则每组包含两个 Buddy 块。在每组内，两个 Buddy 块的地址仅在第 $n$ 位不同：

- 块 A：地址二进制表示为 `...0xxxxxx`
- 块 B：地址二进制表示为 `...1xxxxxx`

因此，通过异或 $2^n$（即第 $n$ 位为 1，其余为 0），即可在两个 Buddy 之间切换。

#### 实际示例：
- **4KB 页（n=0）**：页帧 0x1000 的 Buddy = `0x1000 ^ 0x1000 = 0x0000`
- **8KB 块（n=1，2 页）**：起始页帧 0x2000 的 Buddy = `0x2000 ^ 0x2000 = 0x0000`
- **16KB 块（n=2，4 页）**：起始页帧 0x4000 的 Buddy = `0x4000 ^ 0x4000 = 0x0000`

这种地址算术的优势在于**O(1) 时间复杂度**，无需遍历或搜索。

### 1.2 空闲块管理：位图 vs 空闲链表

Buddy 系统需要跟踪每个空闲块的状态。主要有两种实现方式：

#### 位图（Bitmap）方案
- **原理**：每个块对应位图中的 1 位，1=空闲，0=已分配
- **优点**：内存占用极小（1 bit/块）
- **缺点**：合并时需遍历位图查找 Buddy 状态，时间复杂度 O(k)（k 为位图大小）
- **适用场景**：内存极度受限的嵌入式系统

#### 空闲链表（Free List）方案
- **原理**：每个 order 维护一个空闲块链表，块内嵌链表指针
- **优点**：合并/分裂 O(1) 时间复杂度
- **缺点**：每个块需额外 8 字节（64 位系统）存储指针
- **适用场景**：通用操作系统（Linux 采用此方案）

Linux 采用了**混合策略**：使用 `struct page` 数组跟踪每个页的状态（包含位图信息），同时维护空闲链表用于快速分配。

### 1.3 Linux 内核的 Buddy 数据结构

Linux 将物理内存划分为 **内存区域 **(Zones)，每个 Zone 独立管理 Buddy 系统。关键数据结构如下：

```c
// mmzone.h
struct free_area {
    struct list_head free_list[MIGRATE_TYPES]; // 按迁移类型分离的空闲链表
    unsigned long nr_free;                      // 空闲页数
};

struct zone {
    // 基本信息
    unsigned long watermark[NR_WMARK];          // 水位线
    unsigned long managed_pages;                // 可管理页数
    
    // Buddy 核心
    struct free_area free_area[MAX_ORDER];      // MAX_ORDER=11 (2MB)
    
    // 每 CPU 缓存
    struct per_cpu_pageset __percpu *pageset;
    
    // 锁
    spinlock_t lock;
    
    // 其他字段...
};
```

其中 `MAX_ORDER` 定义为 11，支持最大 $2^{11} = 2048$ 个页（8MB，4KB 页大小）。

### 1.4 极简 Buddy 实现框架

以下是 Buddy 系统的核心算法实现（50 行以内）：

```c
#define MAX_ORDER 10
struct list_head free_list[MAX_ORDER];

void buddy_init() {
    for (int i = 0; i <= MAX_ORDER; i++)
        INIT_LIST_HEAD(&free_list[i]);
}

// 分裂大块到目标 order
void buddy_split(int from_order, int to_order) {
    while (from_order > to_order) {
        from_order--;
        uint32_t buddy = current_block ^ (1 << from_order);
        list_add(&buddy_list, &free_list[from_order]);
    }
}

// 分配
uint32_t buddy_alloc(int order) {
    if (!list_empty(&free_list[order]))
        return get_free_block(order);
    
    for (int i = order + 1; i <= MAX_ORDER; i++) {
        if (!list_empty(&free_list[i])) {
            uint32_t block = get_free_block(i);
            buddy_split(i, order);
            return block;
        }
    }
    return 0; // 失败
}

// 合并
void buddy_merge(uint32_t block, int order) {
    while (order < MAX_ORDER) {
        uint32_t buddy = block ^ (1 << order);
        if (!is_buddy_free(buddy, order)) break;
        
        remove_buddy(buddy, order);
        block = min(block, buddy);
        order++;
    }
    list_add(block, &free_list[order]);
}
```

这个框架包含了 Buddy 系统的所有核心逻辑，但缺少工业级实现的关键优化。

---

## 第二章：水位线与内存保护机制

### 2.1 内存耗尽的死锁风险

在内核执行关键操作时（如处理 Page Fault），可能需要分配内存。如果此时系统内存耗尽：
1. 分配失败 → 触发内存回收
2. 内存回收需要分配临时数据结构 → 再次分配失败
3. 形成死锁循环

**解决方案**：保留一部分“应急内存”（Emergency Memory），仅供关键路径使用。

### 2.2 三级水位线模型

Linux 定义了三级水位线，形成内存使用的安全边界：

| 水位线 | 触发条件 | 处理动作 |
|--------|----------|----------|
| **high** | 空闲页 < high | 唤醒 kswapd 后台回收线程 |
| **low** | 空闲页 < low | 直接回收（阻塞当前进程） |
| **min** | 空闲页 < min | 仅允许 `PF_MEMALLOC` 分配 |

水位线的计算基于 Zone 的总页数：
```c
// mm/page_alloc.c
static void __setup_watermark_min(struct zone *zone)
{
    unsigned long pages = zone_managed_pages(zone);
    
    // min 水位线 = 总页数 / 1024，但至少 8 页
    zone->watermark[WMARK_MIN] = min_wmark_pages(zone);
    
    // low = min * 2, high = min * 3
    zone->watermark[WMARK_LOW]  = min_wmark_pages(zone) * 2;
    zone->watermark[WMARK_HIGH] = min_wmark_pages(zone) * 3;
}
```

### 2.3 内存分配时的水位线检查

每次分配内存时，Buddy 系统会检查水位线：

```c
// mm/page_alloc.c
bool __zone_watermark_ok(struct zone *z, unsigned int order,
                        unsigned long alloc_flags, int classzone_idx,
                        long min)
{
    long free_pages = zone_page_state(z, NR_FREE_PAGES);
    
    // 检查：空闲页 > min + 2^order
    if (free_pages <= min + (1UL << order))
        return false;
    
    // 检查高阶分配的碎片情况
    if (order && !zone_watermark_ok_safe(z, order, alloc_flags))
        return false;
    
    return true;
}
```

### 2.4 kswapd 后台回收线程

kswapd 是 Linux 的内存回收守护进程，其工作流程：

1. **监控水位线**：定期检查所有 Zone 的空闲页数
2. **触发条件**：空闲页 < high 水位线
3. **回收策略**：
   - 优先回收可回收页（page cache）
   - 若仍不足，回收匿名页（swap out）
4. **退出条件**：空闲页 > high 水位线

kswapd 的优势在于**异步回收**，避免阻塞应用程序。

### 2.5 直接回收（Direct Reclaim）

当 kswapd 无法及时回收内存时（空闲页 < low），分配路径会触发直接回收：

```c
// mm/page_alloc.c
static inline bool
should_reclaim_pages(struct zone *zone, gfp_t gfp_mask)
{
    return !zone_watermark_ok(zone, 0, gfp_mask, 0, ALLOC_WMARK_LOW);
}

struct page *alloc_pages_direct_reclaim(gfp_t gfp_mask, unsigned int order)
{
    // 尝试分配
    struct page *page = get_page_from_freelist(gfp_mask, order);
    if (page)
        return page;
    
    // 触发直接回收
    if (should_reclaim_pages(zone, gfp_mask)) {
        shrink_zones(gfp_mask, order); // 回收内存
        page = get_page_from_freelist(gfp_mask, order);
    }
    
    return page;
}
```

直接回收会**阻塞当前进程**，因此是性能敏感路径。

### 2.6 应急分配路径（PF_MEMALLOC）

对于内核关键路径（如内存回收本身），Linux 提供了 `PF_MEMALLOC` 标志：

```c
// 分配内存时设置标志
current->flags |= PF_MEMALLOC;
page = alloc_page(GFP_ATOMIC);
current->flags &= ~PF_MEMALLOC;
```

拥有 `PF_MEMALLOC` 的进程可以：
- 分配低于 min 水位线的内存
- 跳过某些回收检查
- 避免死锁

---

## 第三章：迁移类型与碎片对抗

### 3.1 内存碎片的根本原因

内存碎片分为两类：
- **内部碎片**：分配块大于请求大小（Buddy 天然存在）
- **外部碎片**：空闲内存总量足够，但无法满足大块请求

**Buddy 系统无法解决外部碎片**！因为：
- 不可移动页（如内核代码）固定在特定位置
- 长期运行后，内存被“钉住”，无法合并大块

### 3.2 迁移类型（Migratetype）设计

Linux 通过**迁移类型**将页面分类，隔离不同移动性的页面：

```c
// mmzone.h
enum migratetype {
    MIGRATE_UNMOVABLE,    // 不可移动（内核）
    MIGRATE_MOVABLE,      // 可移动（用户页）
    MIGRATE_RECLAIMABLE,  // 可回收（page cache）
    MIGRATE_HIGHATOMIC,   // 原子分配
    MIGRATE_CMA,          // 连续内存区
    MIGRATE_ISOLATE,      // 隔离页
    MIGRATE_TYPES
};
```

#### 分配时指定类型：
```c
// GFP 标志决定迁移类型
alloc_pages(GFP_KERNEL, order);      // → MIGRATE_MOVABLE
alloc_pages(GFP_ATOMIC, order);      // → MIGRATE_HIGHATOMIC
alloc_pages(GFP_KERNEL | __GFP_HIGH, order); // → MIGRATE_UNMOVABLE
```

### 3.3 空闲链表按类型分离

每个 `free_area[order]` 包含多个空闲链表：

```c
struct free_area {
    struct list_head free_list[MIGRATE_TYPES];
    unsigned long nr_free;
};
```

分配时优先从目标类型的链表获取：
```c
// mm/page_alloc.c
static struct page *get_page_from_free_area(struct free_area *area,
                                           int migratetype)
{
    return list_first_entry_or_null(&area->free_list[migratetype],
                                   struct page, lru);
}
```

### 3.4 Fallback 机制

当目标类型无内存时，Buddy 系统按预定义顺序从其他类型借用：

```c
// mm/page_alloc.c
static int fallbacks[MIGRATE_TYPES][MIGRATE_TYPES-1] = {
    [MIGRATE_UNMOVABLE]   = { MIGRATE_RECLAIMABLE, MIGRATE_MOVABLE, 
                              MIGRATE_HIGHATOMIC, MIGRATE_CMA },
    [MIGRATE_MOVABLE]     = { MIGRATE_RECLAIMABLE, MIGRATE_UNMOVABLE, 
                              MIGRATE_HIGHATOMIC, MIGRATE_CMA },
    [MIGRATE_RECLAIMABLE] = { MIGRATE_MOVABLE, MIGRATE_UNMOVABLE, 
                              MIGRATE_HIGHATOMIC, MIGRATE_CMA },
    // ...
};
```

Fallback 顺序的设计原则：
1. **优先可回收类型**（RECLAIMABLE）
2. **避免污染不可移动类型**
3. **CMA 作为最后选择**

### 3.5 页面迁移（Compaction）

对于可移动页面，Linux 提供**页面迁移**机制，主动整理内存：

1. **扫描**：查找可移动页和空闲区域
2. **迁移**：将可移动页复制到空闲区域
3. **释放**：原区域合并为大块

页面迁移由 `compaction` 子系统实现，可通过 `/proc/sys/vm/compact_unevictable_allowed` 控制。

### 3.6 CMA（连续内存区）实现

CMA 用于需要大块连续物理内存的场景（如 GPU、摄像头）：

```c
// 预留 CMA 区域
struct cma *cma = cma_alloc("gpu", size, order, false);

// 分配连续内存
void *buf = cma_alloc(cma, count, align);
```

CMA 的关键设计：
- **启动时预留**：在 Buddy 初始化前保留区域
- **按需分配**：平时作为普通内存使用，需要时迁移可移动页
- **专用迁移类型**：`MIGRATE_CMA` 隔离 CMA 页面

---

## 第四章：工业级优化与 NUMA

### 4.1 每 CPU 页框缓存（PCP）

#### 问题：全局锁竞争
多核系统中，所有 CPU 共享 `zone->lock`，导致严重竞争。

#### 解决方案：Per-CPU 缓存
每个 CPU 维护私有页框缓存，分配时无锁：

```c
// mmzone.h
struct per_cpu_pages {
    int count;          // 缓存页数
    int high;           // 高水位（触发 drain）
    int batch;          // 批量操作数
    struct list_head list;
};

struct per_cpu_pageset {
    struct per_cpu_pages pcp[2]; // 0=high, 1=low
};
```

#### 分配流程：
1. **检查本地缓存**：`pcp->count > 0` → 直接返回
2. **批量 refill**：缓存空时，从 Buddy 批量分配 `batch` 页
3. **批量 drain**：缓存满时，批量归还 Buddy

#### 性能优势：
- **90%+ 分配无锁**
- **减少 TLB 刷新**（本地内存）
- **提升缓存局部性**

### 4.2 复合页（Compound Page）与大页支持

#### 问题：4KB 页的局限
- **TLB 压力**：大内存应用需大量 TLB 条目
- **页表开销**：4KB 页需 4 级页表（x86_64）

#### 解决方案：复合页
将多个连续页组合为逻辑大页：

```c
// 首页面（存元数据）
struct page {
    unsigned long flags;
    union {
        struct {
            unsigned long compound_head;
            unsigned char compound_order; // log2(页数)
            atomic_t compound_mapcount;
        };
        // ...
    };
};

// 尾页面（指回首页面）
struct page {
    unsigned long flags;
    struct page *compound_head;
};
```

#### 透明大页（THP）自动合并：
```c
// mm/huge_memory.c
static void collapse_huge_page(struct mm_struct *mm, 
                              unsigned long address)
{
    // 1. 检查 512 个 4KB 页是否连续且可合并
    // 2. 分配 2MB 大页
    // 3. 复制内容
    // 4. 更新页表
    // 5. 释放原 4KB 页
}
```

THP 由 khugepaged 内核线程定期扫描触发。

### 4.3 NUMA 感知内存管理

#### NUMA 架构挑战
- **本地节点**：CPU 访问本地内存延迟低
- **远程节点**：跨节点访问延迟高（2-3 倍）

#### Linux NUMA 策略：
1. **本地优先**：`alloc_pages()` 优先从本地节点分配
2. **内存策略**：
   - `MPOL_DEFAULT`：默认策略
   - `MPOL_BIND`：绑定到指定节点
   - `MPOL_INTERLEAVE`：交错分配
3. **自动迁移**：`numa_balancing` 自动迁移热点页面

#### 数据结构：
```c
// mmzone.h
typedef struct pglist_data {
    struct zone node_zones[MAX_NR_ZONES];
    struct zonelist node_zonelists[MAX_ZONELISTS];
    int nr_zones;
    // ...
} pg_data_t;

// 每个 NUMA 节点一个 pg_data_t
extern pg_data_t node_data[];
```

分配时选择节点：
```c
// mm/page_alloc.c
struct page *alloc_pages_node(int nid, gfp_t gfp_mask, unsigned int order)
{
    if (nid == NUMA_NO_NODE)
        nid = numa_mem_id(); // 本地节点
    
    return __alloc_pages(gfp_mask, order, nid);
}
```

### 4.4 工业级 Buddy 实现框架

综合上述所有特性，Buddy 系统的完整框架如下：

```c
// 内存区域（Zone）
struct zone {
    char *name;
    unsigned long managed_pages;
    unsigned long watermark[NR_WMARK];
    
    // Buddy 核心
    struct free_area free_area[MAX_ORDER];
    
    // 每 CPU 缓存
    struct per_cpu_pageset __percpu *pageset;
    
    spinlock_t lock;
};

// 分配入口
struct page *__alloc_pages(gfp_t gfp_mask, unsigned int order, int nid)
{
    // 1. 选择内存区域
    struct zonelist *zonelist = node_zonelist(nid, gfp_mask);
    
    // 2. 遍历区域列表
    for (struct zoneref *z = zonelist->_zonerefs; z->zone; z++) {
        struct zone *zone = z->zone;
        
        // 3. 检查水位线
        if (!zone_watermark_ok(zone, order, gfp_mask))
            continue;
        
        // 4. 从 PCP 分配
        struct page *page = rmqueue_pcpercpu(zone, order, gfp_mask);
        if (page) return page;
        
        // 5. 从 Buddy 分配
        page = rmqueue(zone, order, gfp_mask);
        if (page) return page;
    }
    
    // 6. 触发内存回收
    return __alloc_pages_slowpath(gfp_mask, order, nid);
}

// 回收入口
void __free_pages(struct page *page, unsigned int order)
{
    // 1. 处理复合页
    if (PageCompound(page)) {
        destroy_compound_page(page, order);
        return;
    }
    
    // 2. 尝试放入 PCP
    if (free_pages_prepare(page, order)) {
        free_pcppages_bulk(zone, 1, page);
        return;
    }
    
    // 3. 归还 Buddy
    free_one_page(zone, page, order, migratetype);
}
```

---

## 结论：Buddy 系统的工程艺术

Buddy 分配器从简单的分裂/合并算法，演进为包含**水位线保护、迁移类型隔离、每 CPU 缓存、复合页支持、NUMA 感知**的复杂系统，体现了操作系统内核开发的工程艺术：

1. **理论与实践的结合**：地址算术的数学之美 + 实际场景的工程妥协
2. **性能与安全的平衡**：水位线防止死锁 + PCP 提升多核性能
3. **通用与专用的统一**：通用 Buddy 框架 + CMA/THP 专用优化
4. **简单与复杂的演进**：从 50 行核心代码到 10,000+ 行工业实现

理解 Buddy 系统不仅有助于操作系统开发，更能培养**系统级思维**：在资源约束下，如何通过分层抽象、策略分离、渐进优化，构建既高效又可靠的复杂系统。

对于希望深入 Linux 内核的开发者，建议：
- 阅读 `mm/page_alloc.c` 源码
- 使用 `perf` 和 `trace-cmd` 分析内存分配路径
- 通过 `/proc/buddyinfo` 监控 Buddy 状态
- 实验不同水位线和迁移类型的影响

Buddy 系统的故事，远未结束。随着 CXL、持久内存等新硬件的出现，内存管理将面临新的挑战与机遇。

---

## 附录：关键数据结构与函数速查

### 核心数据结构
| 结构 | 作用 | 文件 |
|------|------|------|
| `struct zone` | 内存区域 | `mmzone.h` |
| `struct free_area` | 空闲块管理 | `mmzone.h` |
| `struct per_cpu_pages` | 每 CPU 缓存 | `mmzone.h` |
| `struct page` | 页描述符 | `mm_types.h` |

### 关键函数
| 函数 | 功能 | 文件 |
|------|------|------|
| `__alloc_pages` | 内存分配入口 | `page_alloc.c` |
| `__free_pages` | 内存回收入口 | `page_alloc.c` |
| `zone_watermark_ok` | 水位线检查 | `page_alloc.c` |
| `rmqueue` | 从 Buddy 分配 | `page_alloc.c` |
| `free_one_page` | 归还 Buddy | `page_alloc.c` |

### 调试接口
| 接口 | 用途 |
|------|------|
| `/proc/buddyinfo` | 查看 Buddy 空闲块 |
| `/proc/zoneinfo` | 查看 Zone 详细信息 |
| `/proc/pagetypeinfo` | 查看迁移类型分布 |
| `echo 1 > /proc/sys/vm/compact_memory` | 手动触发内存整理 |

> **注**：本文基于 Linux 5.15 内核源码分析，具体实现可能因版本而异。