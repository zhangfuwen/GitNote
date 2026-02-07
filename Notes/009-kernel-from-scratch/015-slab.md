
# 从零写 OS 内核-第十五篇：Slab 分配器 —— 高效管理内核小对象

> **“Buddy 系统擅长分配大块内存，但内核频繁申请 32 字节、128 字节的小对象怎么办？  
> 今天，我们实现 Slab 分配器，让内核内存分配快如闪电！”**

在上一篇中，我们构建了完整的内存管理体系：  
- **Buddy 系统**管理物理页（4KB 粒度）  
- **用户态 malloc** 通过 `brk`/`mmap` 申请内存  

但内核自身也有大量**小对象**需要频繁分配/释放：  
✅ **进程控制块（PCB）**  
✅ **文件描述符（file）**  
✅ **VFS 目录项（dentry）**  
✅ **网络缓冲区（skb）**  

如果每次用 `kmalloc` 都向 Buddy 申请 4KB 再切一小块，**内部碎片会高达 99%**！  

今天，我们就来实现 **Slab 分配器**——专为内核小对象设计的高效内存池，  
让你的内核分配 PCB 时**零碎片、零初始化开销**！

---

## 🧱 一、为什么需要 Slab？Buddy 的痛点

### Buddy 系统的局限：
| 场景 | 问题 |
|------|------|
| **分配 32 字节** | Buddy 返回 4KB 页，浪费 99.2% |
| **频繁分配/释放** | 每次都要清零内存（安全要求） |
| **缓存局部性差** | 对象分散在不同页，CPU 缓存命中率低 |

### Slab 的核心思想：
> **“预分配一批同类型对象，用时即取，废时即还”**

- **Slab**：一个或多个连续物理页，划分为固定大小的对象
- **Cache**：管理同类型对象的 Slab 池（如 `task_cache`、`file_cache`）
- **对象复用**：释放的对象不归还 Buddy，而是放入空闲链表，下次直接复用

> 💡 **Slab = 对象缓存 + 空闲链表 + 构造/析构回调**

---

## 📦 二、Slab 核心数据结构

### 1. **kmem_cache（缓存描述符）**
```c
struct kmem_cache {
    char name[32];          // 缓存名（如 "task_struct"）
    size_t obj_size;        // 对象大小
    size_t align;           // 对齐要求
    size_t offset;          // 空闲指针偏移（嵌入对象内部）
    
    // 构造/析构回调
    void (*ctor)(void *obj);
    void (*dtor)(void *obj);
    
    struct list_head slabs_full;    // 完全分配的 slab
    struct list_head slabs_partial; // 部分分配的 slab
    struct list_head slabs_free;    // 完全空闲的 slab
};
```

### 2. **slab（内存页描述符）**
```c
struct slab {
    struct kmem_cache *cache;   // 所属缓存
    void *pages;                // 物理页起始地址
    unsigned int num_objs;      // 总对象数
    unsigned int free_objs;     // 空闲对象数
    void *first_free;           // 第一个空闲对象（嵌入式链表）
    
    struct list_head list;      // 挂在 cache 的 full/partial/free 链表
};
```

> 🔑 **关键技巧**：空闲对象的指针**直接存放在对象内存的开头**（节省额外内存）！

---

## ⚙️ 三、Slab 操作流程

### 1. **创建缓存（kmem_cache_create）**
```c
struct kmem_cache *kmem_cache_create(
    const char *name, 
    size_t size, 
    size_t align,
    void (*ctor)(void*),
    void (*dtor)(void*)
) {
    struct kmem_cache *cache = kmalloc(sizeof(struct kmem_cache));
    strcpy(cache->name, name);
    cache->obj_size = size;
    cache->align = align ? align : sizeof(void*);
    cache->offset = 0; // 空闲指针放在对象开头
    cache->ctor = ctor;
    cache->dtor = dtor;
    
    INIT_LIST_HEAD(&cache->slabs_full);
    INIT_LIST_HEAD(&cache->slabs_partial);
    INIT_LIST_HEAD(&cache->slabs_free);
    
    return cache;
}
```

### 2. **分配对象（kmem_cache_alloc）**
```c
void *kmem_cache_alloc(struct kmem_cache *cache) {
    struct slab *slab;
    void *obj;

    // 1. 优先从 partial slab 分配
    if (!list_empty(&cache->slabs_partial)) {
        slab = list_first_entry(&cache->slabs_partial, struct slab, list);
    } 
    // 2. 其次从 free slab 分配
    else if (!list_empty(&cache->slabs_free)) {
        slab = list_first_entry(&cache->slabs_free, struct slab, list);
    } 
    // 3. 没有可用 slab，新建一个
    else {
        slab = allocate_new_slab(cache);
        if (!slab) return NULL;
        list_add(&slab->list, &cache->slabs_partial);
    }

    // 4. 从 slab 取第一个空闲对象
    obj = slab->first_free;
    slab->first_free = *(void**)obj; // 空闲链表指针存在对象开头
    slab->free_objs--;

    // 5. 如果 slab 变满，移到 full 链表
    if (slab->free_objs == 0) {
        list_move(&slab->list, &cache->slabs_full);
    }

    // 6. 调用构造函数（如果存在）
    if (cache->ctor) {
        cache->ctor(obj);
    }

    return obj;
}
```

### 3. **释放对象（kmem_cache_free）**
```c
void kmem_cache_free(struct kmem_cache *cache, void *obj) {
    struct slab *slab = get_slab_from_obj(obj, cache);
    
    // 1. 调用析构函数
    if (cache->dtor) {
        cache->dtor(obj);
    }
    
    // 2. 将对象放回空闲链表
    *(void**)obj = slab->first_free;
    slab->first_free = obj;
    slab->free_objs++;
    
    // 3. 调整 slab 所在链表
    if (slab->free_objs == 1) {
        // 从 full 移到 partial
        list_move(&slab->list, &cache->slabs_partial);
    } else if (slab->free_objs == slab->num_objs) {
        // 从 partial 移到 free
        list_move(&slab->list, &cache->slabs_free);
    }
}
```

---

## 🏗️ 四、初始化内核关键缓存

在内核启动早期，创建常用对象的缓存：

```c
void slab_init() {
    // 进程控制块缓存
    task_cache = kmem_cache_create(
        "task_struct", 
        sizeof(task_t), 
        0, 
        task_ctor, 
        NULL
    );
    
    // 文件对象缓存
    file_cache = kmem_cache_create(
        "file", 
        sizeof(struct file), 
        0, 
        file_ctor, 
        NULL
    );
    
    // inode 缓存
    inode_cache = kmem_cache_create(
        "inode", 
        sizeof(struct inode), 
        0, 
        inode_ctor, 
        NULL
    );
}
```

### 构造函数示例（初始化 PCB）：
```c
void task_ctor(void *obj) {
    task_t *task = (task_t*)obj;
    memset(task, 0, sizeof(task_t));
    task->state = TASK_RUNNING;
    // 初始化内核栈、链表头等
}
```

> ✅ **从此，创建新进程只需 `kmem_cache_alloc(task_cache)`，无需手动初始化！**

---

## 🔄 五、与 Buddy 系统的协作

Slab 分配器**底层仍依赖 Buddy 系统**获取物理页：

```c
struct slab *allocate_new_slab(struct kmem_cache *cache) {
    // 1. 计算需要多少页
    size_t objs_per_slab = (PAGE_SIZE * pages_per_slab) / cache->obj_size;
    size_t total_size = objs_per_slab * cache->obj_size;
    int order = get_order(total_size);
    
    // 2. 从 Buddy 申请连续页
    void *pages = buddy_alloc(order);
    if (!pages) return NULL;
    
    // 3. 初始化 slab 结构
    struct slab *slab = kmalloc(sizeof(struct slab));
    slab->cache = cache;
    slab->pages = pages;
    slab->num_objs = objs_per_slab;
    slab->free_objs = objs_per_slab;
    
    // 4. 构建空闲链表（对象内部指针）
    void *obj = pages;
    for (int i = 0; i < objs_per_slab - 1; i++) {
        *(void**)obj = (char*)obj + cache->obj_size;
        obj = *(void**)obj;
    }
    *(void**)obj = NULL; // 最后一个
    slab->first_free = pages;
    
    return slab;
}
```

> 🌟 **Slab 是 Buddy 的上层优化，两者协同工作！**

---

## 🧪 六、测试：内核对象分配性能

### 传统方式（无 Slab）：
```c
// 每次分配都要初始化
task_t *task = (task_t*)kmalloc(sizeof(task_t));
memset(task, 0, sizeof(task_t));
task->state = TASK_RUNNING;
// ... 其他初始化
```

### Slab 方式：
```c
task_t *task = kmem_cache_alloc(task_cache);
// 对象已初始化，直接使用！
```

### 性能对比（假设分配 1000 个 PCB）：
| 指标 | 传统方式 | Slab 方式 |
|------|----------|-----------|
| **分配次数** | 1000 次 Buddy 调用 | ~4 次 Buddy 调用（每个 slab 256 个对象）|
| **初始化开销** | 1000 次 memset | 256 次（只在 slab 创建时）|
| **内存碎片** | 高（对象分散） | 低（对象紧凑）|
| **CPU 缓存命中** | 低 | 高（局部性好）|

> ✅ **Slab 在频繁分配小对象时，性能提升 10 倍以上！**

---

## ⚠️ 七、Slab 的进阶优化

1. **着色（Slab Coloring）**  
   - 调整对象起始偏移，避免不同缓存的对象映射到同一 CPU 缓存行  
   - 提升多核性能

2. **硬件缓存对齐**  
   - 确保对象大小是 L1 缓存行（64 字节）的倍数

3. **NUMA 感知**  
   - 在多 CPU 插槽系统中，优先分配本地内存

4. **对象冻结**  
   - 当 slab 完全空闲时，延迟释放物理页（避免频繁 Buddy 调用）

> 💡 **Linux 的 SLAB/SLUB/SLOB 分配器正是基于这些思想演进而来！**

---

## 💬 写在最后

Slab 分配器看似复杂，  
但它解决了内核内存管理的**根本矛盾**：  
**频繁分配小对象 vs 内存效率与性能**。

今天你实现的 `kmem_cache_alloc`，  
正是 Linux 中 `kmalloc` 背后的核心机制之一。

> 🌟 **高效的内存管理，是高性能操作系统的隐形翅膀。**

---

📬 **动手挑战**：  
为你的 ext2 inode 实现 Slab 缓存，并测量创建 100 个文件的性能提升。  
欢迎在评论区分享你的性能对比数据！

👇 下一篇你想看：**写时复制（CoW）实现**，还是 **内存压缩（zswap）**？

---

**#操作系统 #内核开发 #内存管理 #Slab分配器 #kmalloc #内核对象 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“slab”**，获取：
> - 完整 Slab 分配器代码（含 cache/slab 结构）
> - kmem_cache_create/alloc/free 实现
> - 内核关键对象（task/file/inode）缓存初始化模板
