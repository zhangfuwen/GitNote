# 内存管理-内核篇：Slab 分配器深度设计 —— 对比 Linux 实现

> **“Buddy 系统擅长分配大块内存，但内核频繁申请 32 字节的 task_struct 怎么办？  
> 本文将深入 Slab 分配器的设计哲学，对比 Linux 的三种实现（SLAB/SLUB/SLOB），  
> 并构建一个高性能、低碎片的工业级 Slab 系统。”**

## 引言：小对象分配的困境

在操作系统内核中，**小对象**（Small Objects）的分配极其频繁：
- **进程创建**：`task_struct`（约 1.5KB）
- **文件操作**：`file`（约 200 字节）、`dentry`（约 192 字节）
- **网络栈**：`sk_buff`（约 256 字节）
- **VFS 层**：`inode`（约 600 字节）

如果直接使用 Buddy 系统分配这些对象：
- **内部碎片高达 75%+**：分配 4KB 页仅用 200 字节
- **初始化开销巨大**：每次 `memset` 清零整个页
- **缓存局部性差**：对象分散在不同页，CPU 缓存命中率低

**Slab 分配器**正是为解决这些问题而生！它通过**对象缓存 + 空闲链表**，实现：
- **零初始化开销**：对象复用，构造函数按需调用
- **零内部碎片**：精确分配对象大小
- **高缓存命中率**：同类对象紧凑存储

本文将系统性地剖析 Slab 分配器的设计，并**深度对比 Linux 的三种实现**（SLAB/SLUB/SLOB），最后提供一个可运行的工业级框架。

---

## 第一章：Slab 理论模型与核心思想

### 1.1 为什么需要 Slab？

#### Buddy 系统的小对象分配问题：
| 问题 | 后果 | Slab 解决方案 |
|------|------|---------------|
| **内部碎片** | 分配 4KB 页仅用 200 字节，浪费 95% | 精确分配对象大小 |
| **初始化开销** | 每次分配都 `memset` 清零 | 对象复用，按需构造 |
| **缓存局部性** | 对象分散在不同页 | 同类对象紧凑存储 |
| **TLB 压力** | 频繁分配/释放导致 TLB 刷新 | 减少页分配次数 |

#### Slab 的核心思想：
> **“预分配一批同类型对象，用时即取，废时即还”**

- **Slab**：一个或多个连续物理页，划分为固定大小的对象
- **Cache**：管理同类型对象的 Slab 池（如 `task_cache`）
- **对象复用**：释放的对象不归还 Buddy，而是放入空闲链表

### 1.2 Slab 三级架构

Slab 系统采用三级层次结构：

```
+------------------+
|   kmem_cache     | ← 缓存描述符（每种对象一个）
|   - name         |
|   - obj_size     |
|   - ctor/dtor    |
+------------------+
        |
        | 1:N
        v
+------------------+
|      slab        | ← 内存页描述符（每个 slab 一个）
|   - pages        |
|   - free_objs    |
|   - first_free   |
+------------------+
        |
        | 1:N
        v
+------------------+
|     object       | ← 实际对象（嵌入空闲指针）
|   [data][next]   |
+------------------+
```

#### 关键设计决策：
1. **空闲指针内嵌**：空闲对象的指针直接存放在对象内存开头，**零额外内存开销**
2. **构造函数延迟调用**：仅在对象首次分配时调用 `ctor`，复用时不调用
3. **Slab 按需创建**：无空闲对象时才向 Buddy 申请新页

### 1.3 Slab 操作流程

#### 分配对象（kmem_cache_alloc）：
1. 优先从 **partial slab** 分配（有空闲对象）
2. 其次从 **free slab** 分配（全空闲）
3. 若无可用 slab，**新建 slab**（向 Buddy 申请页）
4. 从 slab 取第一个空闲对象
5. **调用构造函数**（如果存在且是首次使用）

#### 释放对象（kmem_cache_free）：
1. **调用析构函数**（如果存在）
2. 将对象放回 slab 的空闲链表
3. 调整 slab 状态（full → partial，partial → free）
4. 若 slab 全空闲且系统内存紧张，**延迟释放**（避免频繁 Buddy 调用）

---

## 第二章：Linux Slab 实现演进史

Linux 内核历史上存在**三种 Slab 实现**，反映了不同设计哲学的演进：

### 2.1 SLAB（经典实现，1990s-2008）

#### 设计特点：
- **复杂但功能完整**：支持对象着色、硬件缓存对齐
- **内存开销大**：每个 slab 需额外元数据（`struct slab`）
- **锁粒度粗**：每个 cache 一个全局锁

#### 核心数据结构：
```c
// mm/slab.c
struct kmem_cache {
    struct array_cache *array[NR_CPUS]; // 每 CPU 缓存
    struct kmem_list3 *nodelists[MAX_NUMNODES];
    unsigned int objsize;               // 对象大小
    unsigned int offset;                // 空闲指针偏移
    void (*ctor)(void *);               // 构造函数
};

struct slab {
    struct list_head list;              // 挂在 full/partial/free 链表
    unsigned long colouroff;            // 着色偏移
    void *s_mem;                        // 第一个对象地址
    unsigned int inuse;                 // 已使用对象数
};
```

#### 优点：
- **对象着色**（Slab Coloring）：避免不同 cache 的对象映射到同一 CPU 缓存行
- **硬件优化**：L1 缓存行对齐

#### 缺点：
- **内存开销大**：元数据占 10-20% 内存
- **复杂度高**：代码超过 10,000 行

### 2.2 SLUB（现代默认，2008-至今）

#### 设计哲学：
> **“简化、去中心化、每 CPU 优化”**

#### 关键改进：
1. **移除 slab 结构**：元数据直接嵌入页描述符（`struct page`）
2. **每 CPU 缓存无锁**：分配/释放完全无锁
3. **简化链表管理**：partial 链表全局共享
4. **动态调试支持**：运行时启用红区（Redzone）、对象跟踪

#### 核心数据结构：
```c
// mm/slub.c
struct kmem_cache {
    struct kmem_cache_cpu __percpu *cpu_slab; // 每 CPU 缓存
    struct kmem_cache_node *node[MAX_NUMNODES];
    unsigned int size;        // 对齐后对象大小
    unsigned int object_size; // 实际对象大小
    unsigned int offset;      // 空闲指针偏移
    void (*ctor)(void *);     // 构造函数
};

// 元数据嵌入 struct page
struct page {
    union {
        struct { 
            unsigned long compound_head;
            unsigned char compound_order;
        };
        struct {
            struct kmem_cache *slab_cache; // 所属 cache
            void *freelist;                // 空闲对象链表
            unsigned int counters;         // inuse + objects
        };
    };
};
```

#### 优势：
- **内存开销极小**：元数据仅 16 字节/页
- **性能卓越**：99% 分配路径无锁
- **调试友好**：`slabinfo`、`/sys/kernel/slab/` 提供详细统计

### 2.3 SLOB（嵌入式简化版）

#### 设计目标：
- **内存极度受限**（<64MB RAM）
- **代码最小化**（<1000 行）

#### 实现原理：
- **简单 first-fit 分配器**
- **无 cache 概念**：所有对象共享一个堆
- **使用 Buddy 页作为堆**

#### 适用场景：
- 嵌入式 Linux（路由器、IoT 设备）
- 内核配置 `CONFIG_SLOB=y`

### 2.4 三种实现对比

| 特性 | SLAB | SLUB | SLOB |
|------|------|------|------|
| **代码复杂度** | 高（10k+ 行） | 中（5k 行） | 低（1k 行） |
| **内存开销** | 高（10-20%） | 低（<5%） | 极低 |
| **性能** | 中 | **高** | 低 |
| **调试支持** | 有限 | **丰富** | 无 |
| **适用场景** | 旧版内核 | **现代默认** | 嵌入式 |

> 💡 **Linux 5.15+ 默认使用 SLUB**，因其在性能、内存、调试三者间取得最佳平衡。

---

## 第三章：工业级 Slab 设计框架

### 3.1 设计目标与约束

#### 核心目标：
1. **高性能**：分配/释放 O(1) 时间，多核无锁
2. **低内存开销**：元数据 < 5% 总内存
3. **高缓存命中率**：同类对象紧凑存储
4. **调试友好**：支持红区、对象跟踪

#### 约束条件：
- **32 位系统**：指针对齐 4 字节
- **4KB 页大小**：slab 大小为页的整数倍
- **无动态重定位**：对象地址固定

### 3.2 数据结构设计

#### 缓存描述符（kmem_cache）
```c
#define KMALLOC_MAX_SIZE 8192

struct kmem_cache {
    char name[32];              // 缓存名（如 "task_struct"）
    size_t obj_size;            // 对象大小
    size_t size;                // 对齐后大小（含空闲指针）
    size_t offset;              // 空闲指针偏移（通常为 0）
    
    // 构造/析构回调
    void (*ctor)(void *obj);
    void (*dtor)(void *obj);
    
    // 每 CPU 缓存（SLUB 风格）
    struct kmem_cache_cpu __percpu *cpu_cache;
    
    // 全局 partial 链表
    struct list_head partial;
    spinlock_t partial_lock;
    
    // 统计信息
    atomic_t total_objects;
    atomic_t active_slabs;
};
```

#### 每 CPU 缓存（kmem_cache_cpu）
```c
struct kmem_cache_cpu {
    void *freelist;             // 本地空闲对象链表
    struct page *page;          // 当前 slab 页
    unsigned int offset;        // 对象偏移
    unsigned int objsize;       // 对象大小
};
```

#### 元数据嵌入页描述符
```c
// 复用 struct page 存储 slab 元数据
struct page {
    union {
        // ... 其他用途
        struct {
            struct kmem_cache *slab_cache; // 所属 cache
            void *freelist;                // 空闲链表
            unsigned int active;           // 活跃对象数
            unsigned int objects;          // 总对象数
        } slab;
    };
};
```

### 3.3 关键算法实现

#### 创建缓存（kmem_cache_create）
```c
struct kmem_cache *kmem_cache_create(const char *name, size_t size,
                                    size_t align, 
                                    void (*ctor)(void*),
                                    void (*dtor)(void*)) {
    struct kmem_cache *cache = kmalloc(sizeof(struct kmem_cache));
    
    // 1. 对齐对象大小
    size = ALIGN(size, align ? align : sizeof(void*));
    
    // 2. 确保能容纳空闲指针
    if (size < sizeof(void*)) size = sizeof(void*);
    
    // 3. 初始化 cache 结构
    strlcpy(cache->name, name, sizeof(cache->name));
    cache->obj_size = size;
    cache->size = size;
    cache->offset = 0; // 空闲指针放在对象开头
    cache->ctor = ctor;
    cache->dtor = dtor;
    
    // 4. 初始化每 CPU 缓存
    cache->cpu_cache = alloc_percpu(struct kmem_cache_cpu);
    
    // 5. 初始化全局链表
    INIT_LIST_HEAD(&cache->partial);
    spin_lock_init(&cache->partial_lock);
    
    return cache;
}
```

#### 分配对象（kmem_cache_alloc）
```c
void *kmem_cache_alloc(struct kmem_cache *cache, gfp_t flags) {
    struct kmem_cache_cpu *c = this_cpu_ptr(cache->cpu_cache);
    void *object;
    
    // 1. 检查本地缓存
    if (likely(c->freelist)) {
        object = c->freelist;
        c->freelist = *(void**)object;
        return object;
    }
    
    // 2. 本地缓存空，从全局 partial 链表获取
    return ___cache_alloc(cache, flags);
}

static void *___cache_alloc(struct kmem_cache *cache, gfp_t flags) {
    struct page *page;
    void *object;
    
    // 1. 从 partial 链表取 slab
    spin_lock(&cache->partial_lock);
    if (!list_empty(&cache->partial)) {
        page = list_first_entry(&cache->partial, struct page, lru);
        list_del(&page->lru);
    } else {
        // 2. 无 partial slab，新建 slab
        page = allocate_slab(cache, flags);
        if (!page) {
            spin_unlock(&cache->partial_lock);
            return NULL;
        }
    }
    spin_unlock(&cache->partial_lock);
    
    // 3. 从 slab 取对象
    object = page->freelist;
    page->freelist = *(void**)object;
    page->active++;
    
    // 4. 更新每 CPU 缓存
    struct kmem_cache_cpu *c = this_cpu_ptr(cache->cpu_cache);
    c->page = page;
    c->freelist = page->freelist;
    c->objsize = cache->obj_size;
    
    // 5. 调用构造函数（首次使用）
    if (unlikely(cache->ctor)) {
        cache->ctor(object);
    }
    
    return object;
}
```

#### 释放对象（kmem_cache_free）
```c
void kmem_cache_free(struct kmem_cache *cache, void *object) {
    struct kmem_cache_cpu *c = this_cpu_ptr(cache->cpu_cache);
    struct page *page;
    
    // 1. 调用析构函数
    if (unlikely(cache->dtor)) {
        cache->dtor(object);
    }
    
    // 2. 检查是否属于当前 slab
    if (likely(page_to_nid(virt_to_page(object)) == numa_node_id() &&
               c->page == virt_to_page(object))) {
        // 本地释放：无锁
        *(void**)object = c->freelist;
        c->freelist = object;
        return;
    }
    
    // 3. 远程释放：需锁
    ___cache_free(cache, object);
}

static void ___cache_free(struct kmem_cache *cache, void *object) {
    struct page *page = virt_to_page(object);
    
    // 1. 放回 slab 空闲链表
    *(void**)object = page->freelist;
    page->freelist = object;
    page->active--;
    
    // 2. 检查 slab 状态
    if (page->active == 0) {
        // slab 全空闲，释放物理页
        free_slab(cache, page);
    } else if (page->active == 1) {
        // slab 从 full 变 partial，加入全局链表
        spin_lock(&cache->partial_lock);
        list_add(&page->lru, &cache->partial);
        spin_unlock(&cache->partial_lock);
    }
}
```

### 3.4 slab 创建与销毁

#### 分配新 slab
```c
static struct page *allocate_slab(struct kmem_cache *cache, gfp_t flags) {
    // 1. 计算 slab 大小（页数）
    size_t objs_per_slab = (PAGE_SIZE * pages_per_slab) / cache->size;
    int order = get_order(objs_per_slab * cache->size);
    
    // 2. 从 Buddy 分配连续页
    struct page *page = alloc_pages(flags, order);
    if (!page) return NULL;
    
    // 3. 初始化 slab 元数据
    page->slab_cache = cache;
    page->active = 0;
    page->objects = objs_per_slab;
    
    // 4. 构建空闲链表（内嵌指针）
    void *obj = page_address(page);
    void *last = obj;
    for (int i = 0; i < objs_per_slab - 1; i++) {
        *(void**)obj = (char*)obj + cache->size;
        last = obj;
        obj = *(void**)obj;
    }
    *(void**)last = NULL; // 最后一个
    page->freelist = page_address(page);
    
    return page;
}
```

#### 释放 slab
```c
static void free_slab(struct kmem_cache *cache, struct page *page) {
    // 1. 从 partial 链表移除
    spin_lock(&cache->partial_lock);
    list_del(&page->lru);
    spin_unlock(&cache->partial_lock);
    
    // 2. 释放物理页
    __free_pages(page, compound_order(page));
}
```

---

## 第四章：kmalloc 接口与多级缓存

### 4.1 kmalloc 的设计挑战

`kmalloc` 需要支持任意大小（8-8192 字节）的分配，但：
- **不能为每个大小创建 cache**（缓存爆炸）
- **需要快速映射大小到 cache**

#### 解决方案：**多级大小类**（Size Classes）

Linux 将分配大小分为**离散的大小类**，每个类对应一个 cache：

| 大小范围 | Cache 名称 | 实际分配大小 |
|----------|------------|--------------|
| 8-16     | kmalloc-16 | 16           |
| 17-32    | kmalloc-32 | 32           |
| 33-64    | kmalloc-64 | 64           |
| ...      | ...        | ...          |
| 4097-8192| kmalloc-8k | 8192         |

#### 大小类计算：
```c
// mm/slab_common.c
static size_t kmalloc_size_roundup(size_t size) {
    if (size <= 192) {
        // 8-192: 8/16/24/.../192
        return (size + 7) & ~7;
    } else if (size <= 8192) {
        // 256-8192: 256/512/1024/.../8192
        return 1 << (fls(size - 1));
    }
    return size;
}
```

### 4.2 kmalloc 实现框架

#### 初始化 kmalloc caches
```c
// mm/slab_common.c
static struct kmem_cache *kmalloc_caches[KMALLOC_SHIFT_MAX];

void init_kmalloc_caches(void) {
    // 1. 创建通用 caches
    for (int i = KMALLOC_SHIFT_LOW; i <= KMALLOC_SHIFT_MAX; i++) {
        char name[32];
        snprintf(name, sizeof(name), "kmalloc-%d", 1 << i);
        kmalloc_caches[i] = kmem_cache_create(name, 1 << i, 0, NULL, NULL);
    }
    
    // 2. 创建 DMA caches（可选）
    // ...
}
```

#### kmalloc/kfree 接口
```c
void *kmalloc(size_t size, gfp_t flags) {
    if (size <= 192) {
        // 小对象：8/16/24/.../192
        int index = (size + 7) >> 3; // 8-byte aligned
        return kmem_cache_alloc(kmalloc_caches[index], flags);
    } else if (size <= 8192) {
        // 大对象：256/512/.../8192
        int order = fls(size - 1);
        return kmem_cache_alloc(kmalloc_caches[order], flags);
    } else {
        // 超大对象：直接 Buddy
        return (void*)__get_free_pages(flags, get_order(size));
    }
}

void kfree(const void *obj) {
    if (!obj) return;
    
    // 1. 通过地址反查 cache（Linux 用页描述符）
    struct page *page = virt_to_page(obj);
    if (PageSlab(page)) {
        kmem_cache_free(page->slab_cache, (void*)obj);
    } else {
        // Buddy 分配的大块
        __free_pages(page, compound_order(page));
    }
}
```

### 4.3 地址反查 cache 机制

#### 问题：如何从对象地址找到所属 cache？
- **SLAB**：通过 `obj_to_index` 计算 cache
- **SLUB**：**元数据嵌入页描述符**，直接访问

#### SLUB 方案（工业级）：
```c
// mm/slub.c
static struct kmem_cache *slab_pre_alloc_hook(struct kmem_cache *s, gfp_t flags)
{
    // 分配前 hook
}

static void slab_post_alloc_hook(struct kmem_cache *s, gfp_t flags, void *object)
{
    // 分配后 hook（红区检查）
}

// 释放时反查
void kfree(const void *x)
{
    struct page *page;
    
    if (unlikely(ZERO_OR_NULL_PTR(x)))
        return;
    
    page = virt_to_head_page(x);
    if (unlikely(!PageSlab(page))) {
        // Buddy 分配
        __free_pages(page, compound_order(page));
        return;
    }
    
    // 从页描述符获取 cache
    slab_free(page->slab_cache, page, x);
}
```

> ✅ **SLUB 的页描述符嵌入方案是性能关键**！

---

## 第五章：高级优化与调试特性

### 5.1 对象着色（Slab Coloring）

#### 问题：不同 cache 的对象映射到同一 CPU 缓存行
- **后果**：缓存颠簸（Cache Thrashing）
- **解决方案**：**着色**（Coloring）

#### 实现原理：
- 每个 slab 的**起始偏移**不同
- 偏移量 = `color * L1_CACHE_BYTES`
- 确保对象起始地址在不同缓存行

#### 代码框架：
```c
// mm/slab.c (SLAB 实现)
static void *alloc_slab_page(gfp_t flags, int node, struct kmem_cache *cachep)
{
    // 计算着色
    unsigned int color = cachep->color_next;
    cachep->color_next = (color + 1) % cachep->color_off;
    
    // 分配页 + 偏移
    void *addr = alloc_pages(flags, order);
    return addr + color * L1_CACHE_BYTES;
}
```

> 💡 **SLUB 移除了显式着色，依赖硬件预取优化**

### 5.2 红区（Redzone）与调试支持

#### 红区设计：
- 在对象前后添加**保护区**（通常 8-16 字节）
- 填充特定模式（如 `0x5a5a5a5a`）
- 释放时检查是否被覆盖

#### SLUB 调试框架：
```c
// mm/slub.c
#ifdef CONFIG_SLUB_DEBUG
struct track {
    unsigned long addr;
    int cpu;
    int pid;
};

struct kmem_cache {
    unsigned long min_partial;
    unsigned long red_left_pad; // 左红区大小
    // ...
};

static void setup_object_debug(struct kmem_cache *s, struct page *page, void *object)
{
    // 初始化红区
    memset((char*)object - s->red_left_pad, 0x5a, s->red_left_pad);
    memset((char*)object + s->object_size, 0x5a, s->inuse - s->object_size);
    
    // 记录分配/释放轨迹
    set_track(s, object, TRACK_ALLOC);
}
#endif
```

### 5.3 内存泄漏检测

#### 对象跟踪（Object Tracking）：
- 记录每个对象的分配/释放轨迹
- 通过 `/sys/kernel/slab/<cache>/alloc_calls` 查看

#### 实现：
```c
static void set_track(struct kmem_cache *s, void *object, enum track_item alloc)
{
    struct track *p = get_track(s, object, alloc);
    
    p->addr = return_address(0);
    p->cpu = smp_processor_id();
    p->pid = current->pid;
}
```

---

## 结论：Slab 设计的工程权衡

Slab 分配器的设计体现了操作系统内核开发的核心哲学：**在性能、内存、复杂度之间寻找最优平衡**。

- **SLAB** 选择了**功能完整**，但牺牲了内存效率
- **SLUB** 选择了**简化与性能**，成为现代默认
- **SLOB** 选择了**极致精简**，适用于嵌入式

对于我们的实现，**SLUB 风格是最佳选择**：
- **元数据嵌入页描述符**：内存开销最小
- **每 CPU 无锁缓存**：多核性能卓越
- **动态调试支持**：开发友好

理解 Slab 系统不仅有助于内核开发，更能培养**缓存友好编程**（Cache-Friendly Programming）的意识——这是高性能系统软件的基石。

---

## 附录：关键数据结构与函数速查

### 核心数据结构
| 结构 | 作用 | 文件 |
|------|------|------|
| `struct kmem_cache` | 缓存描述符 | `slab.h` |
| `struct kmem_cache_cpu` | 每 CPU 缓存 | `slab.h` |
| `struct page` (slab) | slab 元数据 | `mm_types.h` |

### 关键函数
| 函数 | 功能 | 文件 |
|------|------|------|
| `kmem_cache_create` | 创建缓存 | `slab_common.c` |
| `kmem_cache_alloc` | 分配对象 | `slub.c` |
| `kmem_cache_free` | 释放对象 | `slub.c` |
| `kmalloc`/`kfree` | 通用接口 | `slab_common.c` |

### 调试接口
| 接口 | 用途 |
|------|------|
| `/sys/kernel/slab/` | 查看所有 cache 信息 |
| `slabinfo` | 命令行工具 |
| `echo 1 > /sys/kernel/slab/<cache>/trace` | 启用轨迹跟踪 |
