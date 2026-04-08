
# 内存管理-高级篇：写时复制、交换与 NUMA 优化深度解析

> **“当物理内存不足时，操作系统如何优雅地处理？  
> 本文将深入写时复制（CoW）、交换（Swapping）、NUMA 优化三大高级特性，  
> 并对比 Linux 的工业级实现，构建一个完整的内存过载处理框架。”**

## 引言：内存过载的挑战与解决方案

现代操作系统面临的核心挑战是：**物理内存有限，但应用内存需求无限**。当系统内存压力增大时，内核必须：
- **高效 fork 进程**：避免复制大量内存页
- **优雅处理内存不足**：通过交换释放物理内存
- **优化多核 NUMA 系统**：减少跨节点访问延迟

三大高级内存特性正是为解决这些问题而生：
- **写时复制**（CoW）：`fork` 时共享页，写时才复制
- **交换**（Swapping）：将不活跃页换出到磁盘
- **NUMA 感知**：本地节点优先分配

本文将系统性地剖析这些特性，从算法原理到 Linux 实现，并提供可运行的工业级框架。

---

## 第一章：写时复制（Copy-on-Write, CoW）

### 1.1 CoW 的核心价值

#### fork 的性能问题：
- **传统 fork**：复制父进程所有页 → 内存/时间开销巨大
- **实际场景**：`fork` 后通常立即 `exec`，复制的页被浪费

#### CoW 解决方案：
- **fork 时共享页**：父子进程共享物理页
- **只读保护**：页表项 R/W=0
- **写时复制**：Page Fault 时复制页

#### 性能提升：
- **fork 速度提升 10-100 倍**
- **内存节省 50%+**（共享库、只读数据）

### 1.2 CoW 实现原理

#### fork 时的页表操作
```c
// fork.c
int do_fork() {
    task_t *parent = current_task;
    task_t *child = alloc_task();
    
    // 1. 复制页目录（仅用户空间）
    child->pgdir = copy_page_directory(parent->pgdir);
    
    // 2. 设置共享页为只读
    set_cow_pages(parent->pgdir, child->pgdir);
    
    return child->pid;
}

static void set_cow_pages(pgd_t *parent_pgd, pgd_t *child_pgd) {
    for (int i = 0; i < 768; i++) { // 用户空间 0-767
        if (parent_pgd[i] & PAGE_PRESENT) {
            // 共享页表
            child_pgd[i] = parent_pgd[i];
            
            // 获取页表
            pte_t *pt = (pte_t*)(parent_pgd + (i * 1024));
            for (int j = 0; j < 1024; j++) {
                if (pt[j] & PAGE_PRESENT) {
                    // 设置只读
                    pt[j] &= ~PAGE_RW;
                    // 子进程页表同步
                    ((pte_t*)(child_pgd + (i * 1024)))[j] &= ~PAGE_RW;
                }
            }
        }
    }
}
```

#### Page Fault 处理 CoW
```c
// page_fault.c
void handle_cow_fault(uint32_t fault_addr) {
    task_t *task = current_task;
    uint32_t vaddr = fault_addr & ~0xFFF;
    
    // 1. 获取物理页
    uint32_t paddr = get_phys_addr(task->pgdir, vaddr);
    if (!paddr) return; // 无效地址
    
    // 2. 分配新页
    void *new_page = buddy_alloc(0);
    if (!new_page) {
        send_sigkill(task, SIGKILL);
        return;
    }
    
    // 3. 复制内容
    memcpy(new_page, (void*)(vaddr), PAGE_SIZE);
    
    // 4. 更新页表（可写）
    map_page(task->pgdir, vaddr, (uint32_t)new_page, 
             PAGE_PRESENT | PAGE_RW | PAGE_USER);
    
    // 5. 释放旧页（如果无其他引用）
    // 简化版：假设仅父子进程共享
    // 实际需引用计数
}
```

### 1.3 Linux CoW 实现细节

#### 引用计数管理
Linux 使用 **页描述符**（`struct page`）跟踪引用：
```c
// mm_types.h
struct page {
    atomic_t _refcount; // 引用计数
    // ...
};

// fork 时增加引用
static inline void get_page(struct page *page) {
    atomic_inc(&page->_refcount);
}

// free 时减少引用
static inline void put_page(struct page *page) {
    if (atomic_dec_and_test(&page->_refcount)) {
        __free_pages(page, 0);
    }
}
```

#### 页表项标志
Linux 使用 **特殊标志** 标记 CoW 页：
- **`_PAGE_DIRTY`**：写时置 1，但 CoW 页初始为 0
- **`_PAGE_RW`**：0 表示只读（可能 CoW）

#### 优化：零页（Zero Page）
- **只读零页**：共享的全零页
- **节省内存**：未写入的 CoW 页指向零页

```c
// mm/memory.c
static struct page *zero_page;

static int do_anonymous_page(struct vm_fault *vmf) {
    // 首次访问匿名页 → 映射到 zero_page
    if (!vmf->pte) {
        set_pte(vmf->pte, mk_pte(zero_page, vmf->vma->vm_page_prot));
        return 0;
    }
    // ...
}
```

---

## 第二章：交换（Swapping）机制

### 2.1 交换的核心价值

#### 物理内存不足的后果：
- **OOM Killer**：强制杀死进程
- **系统卡死**：无法分配关键内存

#### 交换解决方案：
- **交换区**（Swap Partition/File）：磁盘上的内存扩展
- **页面换出**：将不活跃页写入交换区
- **页面换入**：访问时从交换区读回

#### 优势：
- **避免 OOM**：内存不足时优雅降级
- **支持大内存应用**：虚拟内存 > 物理内存

### 2.2 交换区管理

#### 交换区数据结构
```c
// swap.h
#define MAX_SWAPFILES 32
#define SWAP_MAP_MAX 0x7fff
#define SWAP_MAP_BAD 0x8000

struct swap_info_struct {
    unsigned char *swap_map;    // 位图：0=空闲，>0=引用计数
    unsigned int flags;         // SWP_ACTIVE, SWP_WRITEOK
    sector_t *swap_file;        // 交换文件/分区
    unsigned int max;           // 最大页数
    unsigned int pages;         // 空闲页数
};

static struct swap_info_struct swap_info[MAX_SWAPFILES];
static int nr_swapfiles;
```

#### 交换区初始化
```c
// swapfile.c
int sys_swapon(const char *specialfile, int swap_flags) {
    // 1. 打开交换文件/分区
    int fd = sys_open(specialfile, O_RDWR, 0);
    struct file *file = get_file(fd);
    
    // 2. 验证交换签名
    char signature[10];
    vfs_pread(file, signature, 10, 0);
    if (memcmp(signature, "SWAP-SPACE", 10) != 0) {
        return -1;
    }
    
    // 3. 初始化 swap_info
    struct swap_info_struct *si = &swap_info[nr_swapfiles++];
    si->swap_map = kmalloc(bitmap_size);
    si->swap_file = file;
    si->max = file_size / PAGE_SIZE;
    si->pages = si->max;
    si->flags = SWP_ACTIVE | SWP_WRITEOK;
    
    return 0;
}
```

### 2.3 页面换出机制

#### 页面换出流程
1. **内存回收触发**：水位线低于 low
2. **选择不活跃页**：LRU 算法
3. **写入交换区**：分配交换槽，写磁盘
4. **更新页表**：标记为交换页

#### 交换槽分配
```c
// swap.c
swp_entry_t get_swap_page(void) {
    for (int i = 0; i < nr_swapfiles; i++) {
        struct swap_info_struct *si = &swap_info[i];
        if (!(si->flags & SWP_WRITEOK)) continue;
        
        // 查找空闲槽
        for (int j = 0; j < si->max; j++) {
            if (si->swap_map[j] == 0) {
                si->swap_map[j] = 1; // 引用计数=1
                si->pages--;
                return swp_entry(i, j);
            }
        }
    }
    return (swp_entry_t){0};
}
```

#### 页表项标记交换页
```c
// 页表项复用：低 3 位存储类型
#define SWP_TYPE(entry) ((entry).val & 0x1f)
#define SWP_OFFSET(entry) ((entry).val >> 5)
#define SWP_ENTRY(type, offset) ((type) | ((offset) << 5))

// 设置交换页表项
void set_swap_pte(pte_t *pte, swp_entry_t entry) {
    *pte = SWP_ENTRY(SWP_TYPE(entry), SWP_OFFSET(entry));
}
```

### 2.4 页面换入机制

#### Page Fault 处理交换页
```c
// page_fault.c
void handle_swap_fault(uint32_t fault_addr, pte_t pte) {
    // 1. 解析交换项
    swp_entry_t entry = { .val = pte };
    
    // 2. 分配物理页
    void *page = buddy_alloc(0);
    if (!page) {
        send_sigkill(current_task, SIGKILL);
        return;
    }
    
    // 3. 从交换区读取
    struct swap_info_struct *si = &swap_info[SWP_TYPE(entry)];
    sector_t offset = SWP_OFFSET(entry) * (PAGE_SIZE / 512);
    block_read(si->swap_file, page, PAGE_SIZE, offset);
    
    // 4. 更新页表
    map_page(current_task->pgdir, fault_addr & ~0xFFF, 
             (uint32_t)page, PAGE_PRESENT | PAGE_RW | PAGE_USER);
    
    // 5. 释放交换槽
    si->swap_map[SWP_OFFSET(entry)] = 0;
    si->pages++;
}
```

### 2.5 Linux 交换实现细节

#### LRU 链表
Linux 使用 **双 LRU 链表** 管理页：
- **活跃链表**（Active）：频繁访问的页
- **不活跃链表**（Inactive）：候选换出页

```c
// mm/swap.c
struct lruvec {
    struct list_head lists[NR_LRU_LISTS];
};

enum lru_list {
    LRU_INACTIVE_ANON = 0,
    LRU_ACTIVE_ANON = 1,
    LRU_INACTIVE_FILE = 2,
    LRU_ACTIVE_FILE = 3,
    NR_LRU_LISTS
};
```

#### 页面回收策略
- **kswapd**：后台回收，维持水位线
- **直接回收**：分配路径触发
- **内存压缩**（zswap）：压缩页而非换出

---

## 第三章：NUMA 感知内存管理

### 3.1 NUMA 架构挑战

#### NUMA 基础：
- **本地节点**：CPU 访问本地内存延迟低
- **远程节点**：跨节点访问延迟高（2-3 倍）

#### 问题：
- **随机分配**：50% 内存访问跨节点
- **性能下降**：应用性能降低 20-50%

### 3.2 NUMA 内存分配策略

#### 基本策略：
- **本地优先**：优先分配本地节点内存
- **轮询分配**：多节点间轮询（避免单节点耗尽）
- **绑定分配**：强制指定节点（`MPOL_BIND`）

#### Linux NUMA 策略：
```c
// include/linux/mempolicy.h
enum numa_policy_mode {
    MPOL_DEFAULT,   // 默认策略
    MPOL_PREFERRED, // 首选节点
    MPOL_BIND,      // 绑定节点
    MPOL_INTERLEAVE,// 交错分配
};
```

### 3.3 NUMA 数据结构

#### 节点内存描述符
```c
// mmzone.h
typedef struct pglist_data {
    struct zone node_zones[MAX_NR_ZONES];
    struct zonelist node_zonelists[MAX_ZONELISTS];
    int nr_zones;
    nodemask_t node_spanned_pages;
    // ...
} pg_data_t;

extern pg_data_t node_data[MAX_NUMNODES];
```

#### 内存策略
```c
// mempolicy.h
struct mempolicy {
    atomic_t refcnt;
    unsigned short mode;
    unsigned short flags;
    union {
        nodemask_t nodes;     // 节点位图
        struct {
            int preferred_node;
        } preferred;
    } v;
};
```

### 3.4 NUMA 分配实现

#### 节点选择
```c
// page_alloc.c
int numa_mem_id(void) {
    return cpu_to_node(smp_processor_id());
}

struct page *alloc_pages_node(int nid, gfp_t gfp_mask, unsigned int order) {
    if (nid == NUMA_NO_NODE)
        nid = numa_mem_id(); // 本地节点
    
    return __alloc_pages(gfp_mask, order, nid);
}
```

#### 交错分配（Interleave）
```c
// mempolicy.c
static int interleave_nodes(struct mempolicy *policy) {
    static unsigned long sequence;
    nodemask_t nodes = policy->v.nodes;
    
    // 轮询选择节点
    int nid = next_node_in(sequence++ % MAX_NUMNODES, nodes);
    return nid;
}
```

### 3.5 页面迁移（Page Migration）

#### 迁移场景：
- **NUMA 平衡**：将远程页迁移到本地
- **内存碎片整理**：合并大块内存

#### 迁移流程：
1. **分配新页**：本地节点
2. **复制内容**：原页 → 新页
3. **更新映射**：所有进程页表
4. **释放原页**

#### Linux 实现：
```c
// mm/migrate.c
int migrate_pages(struct list_head *from, 
                  new_page_t get_new_page, 
                  unsigned long flags) {
    // 1. 分配新页
    // 2. 复制内容
    // 3. 更新页表（通过 rmap）
    // 4. 释放原页
}
```

---

## 第四章：内存压缩（zswap）简介

### 4.1 zswap 的核心思想

#### 传统交换问题：
- **磁盘 I/O 慢**：换出/换入延迟高
- **SSD 寿命**：频繁写入损耗

#### zswap 解决方案：
- **压缩页存储在内存**：避免磁盘 I/O
- **仅当压缩池满时换出**：减少磁盘访问

#### 工作流程：
1. **页面换出**：压缩页存入 zswap 池
2. **池满时**：换出最旧压缩页到交换区
3. **页面换入**：从 zswap 池解压

### 4.2 zswap 数据结构

#### 压缩池
```c
// zswap.h
struct zswap_pool {
    struct crypto_comp *tfm; // 压缩算法
    struct zpool *zpool;     // 内存池
    struct list_head list;
    int refcount;
};

static struct zswap_pool *zswap_pool;
```

#### 压缩页项
```c
struct zswap_entry {
    struct rb_node rbnode;   // 红黑树节点
    swp_entry_t swpentry;    // 交换项
    struct zpool_handle *handle; // 压缩数据句柄
};
```

### 4.3 zswap 性能优势

| 指标 | 传统交换 | zswap |
|------|----------|-------|
| **换出延迟** | 10-100ms | 0.1-1ms |
| **换入延迟** | 10-100ms | 0.1-1ms |
| **SSD 寿命** | 低 | 高 |
| **内存开销** | 无 | 压缩池 |

> 💡 **zswap 适合内存充足但磁盘慢的场景**（如笔记本、虚拟机）

---

## 第五章：Linux 高级内存特性对比

### 5.1 CoW 实现对比

| 特性 | 简化实现 | Linux |
|------|----------|-------|
| **引用计数** | 无 | `struct page->_refcount` |
| **零页优化** | 无 | 共享 zero_page |
| **页表标志** | 手动管理 | `_PAGE_DIRTY` 自动跟踪 |
| **大页 CoW** | 不支持 | 支持 THP CoW |

### 5.2 交换实现对比

| 特性 | 简化实现 | Linux |
|------|----------|-------|
| **LRU 算法** | 无 | 双 LRU 链表 |
| **交换类型** | 仅匿名页 | 匿名页 + 文件页 |
| **压缩** | 无 | zswap/zram |
| **多交换区** | 支持 | 支持，按优先级 |

### 5.3 NUMA 实现对比

| 特性 | 简化实现 | Linux |
|------|----------|-------|
| **策略类型** | 仅本地优先 | MPOL_DEFAULT/INTERLEAVE/BIND |
| **页面迁移** | 无 | 支持 numa_migrate_pages |
| **自动平衡** | 无 | numa_balancing 内核线程 |
| **CPU 绑定** | 无 | cpuset 支持 |

---

## 结论：高级内存管理的系统工程

高级内存特性是操作系统内核的**系统工程典范**，它们融合了：
- **算法优化**（CoW、LRU、NUMA 策略）
- **硬件特性**（TLB、NUMA 拓扑）
- **性能工程**（zswap 压缩、页面迁移）
- **用户体验**（避免 OOM、平滑降级）

理解这些特性不仅有助于内核开发，更能培养**全栈系统思维**：
- **资源约束**：物理内存有限，需智能管理
- **性能权衡**：内存 vs 磁盘 vs CPU
- **架构演进**：从单核到 NUMA，从交换到压缩

对于希望深入 Linux 内核的开发者，建议：
- 阅读 `mm/memory.c`、`mm/swap.c`、`mm/mempolicy.c`
- 使用 `numastat`、`vmstat` 监控内存行为
- 通过 `perf` 分析 Page Fault 和交换开销

高级内存管理的故事，仍在继续。随着 CXL、持久内存等新硬件的出现，这一复杂而优雅的系统将面临新的挑战与机遇。

---

## 附录：关键数据结构与函数速查

### 核心数据结构
| 结构 | 作用 | 文件 |
|------|------|------|
| `struct page` | 页描述符 | mm_types.h |
| `struct swap_info_struct` | 交换区 | swap.h |
| `pg_data_t` | NUMA 节点 | mmzone.h |
| `struct mempolicy` | 内存策略 | mempolicy.h |

### 关键函数
| 函数 | 功能 | 文件 |
|------|------|------|
| `do_cow_fault` | CoW 处理 | memory.c |
| `shrink_lruvec` | 页面回收 | vmscan.c |
| `alloc_pages_node` | NUMA 分配 | page_alloc.c |
| `zswap_shrink` | zswap 压缩池收缩 | zswap.c |

### 调试接口
| 接口 | 用途 |
|------|------|
| `/proc/swaps` | 交换区信息 |
| `/proc/buddyinfo` | Buddy 空闲块 |
| `numastat` | NUMA 内存统计 |
| `cat /sys/kernel/mm/compact_memory` | 手动内存整理 |
