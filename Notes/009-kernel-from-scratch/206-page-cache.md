
# 存储系统-I/O 篇：为自制 OS 实现高效 I/O 路径

> **“write() 调用后，数据何时真正写入磁盘？  
> 本文将深度解析 I/O 路径：从系统调用到 Page Cache，  
> 从回写机制到 I/O 调度器，构建高效、可靠的存储 I/O 栈。”**

## 引言：I/O 路径的核心挑战

在自制操作系统中，实现文件系统只是第一步。  
**如何高效、可靠地处理 I/O**？  
这是存储子系统的终极挑战。

用户调用 `write(fd, buf, 4096)` 后，内核面临关键决策：
- **立即写磁盘**？ → 性能差（每次 I/O）
- **缓存后延迟写**？ → 风险高（断电丢数据）
- **合并相邻请求**？ → 减少磁头移动
- **异步通知完成**？ → 提升应用性能

本文将为自制 OS 设计一个**高效、可配置、安全**的 I/O 路径，  
涵盖缓存、回写、调度、异步等核心机制。

---

## 第一章：I/O 路径核心设计原则

### 1.1 Buffered I/O vs Direct I/O

#### Buffered I/O（默认）：
- **数据写入 Page Cache**
- **延迟写入磁盘**（回写机制）
- **优势**：减少磁盘 I/O，提升性能
- **风险**：断电可能丢数据

#### Direct I/O（O_DIRECT）：
- **绕过 Page Cache**
- **直接写入磁盘**
- **优势**：数据一致性高，适合数据库
- **劣势**：性能可能下降（无缓存）

### 1.2 设计目标

#### 核心目标：
1. **高性能**：Page Cache 减少磁盘 I/O
2. **可靠性**：可控的回写策略，平衡性能与安全
3. **灵活性**：支持 Direct I/O 和 Buffered I/O
4. **可扩展**：预留异步 I/O 接口

#### 约束条件：
- **内存有限**：Page Cache 需 LRU 回收
- **单核设计**：暂不考虑并发控制
- **简单调度**：实现 noop 调度器

---

## 第二章：Page Cache 实现

### 2.1 Page Cache 核心数据结构

```c
// mm/page_cache.h
#define PAGE_CACHE_SHIFT 12
#define PAGE_CACHE_SIZE  (1UL << PAGE_CACHE_SHIFT)

struct page_cache_entry {
    struct vfs_inode *inode;    // 所属 inode
    uint64_t index;             // 页索引（文件内偏移 / PAGE_SIZE）
    void *data;                 // 页数据（4KB）
    uint32_t flags;             // PG_dirty, PG_uptodate
    
    // LRU 链表
    struct list_head lru;
    struct hlist_node hash;     // 哈希表节点
    
    atomic_t count;             // 引用计数
};

// 全局 Page Cache
static struct hlist_head page_cache_hash[PAGE_CACHE_HASH_SIZE];
static LIST_HEAD(page_cache_lru);
static spinlock_t page_cache_lock;
```

### 2.2 页查找与分配

```c
// mm/page_cache.c
struct page_cache_entry *find_or_create_page(struct vfs_inode *inode, 
                                            uint64_t index) {
    uint32_t hash = inode->i_ino ^ (index >> 8) ^ index;
    struct hlist_head *head = &page_cache_hash[hash % PAGE_CACHE_HASH_SIZE];
    
    spin_lock(&page_cache_lock);
    
    // 1. 查找现有页
    struct page_cache_entry *page;
    hlist_for_each_entry(page, head, hash) {
        if (page->inode == inode && page->index == index) {
            atomic_inc(&page->count);
            spin_unlock(&page_cache_lock);
            return page;
        }
    }
    
    // 2. 分配新页
    page = kmalloc(sizeof(struct page_cache_entry));
    page->inode = inode;
    page->index = index;
    page->data = kmalloc(PAGE_CACHE_SIZE);
    page->flags = 0;
    atomic_set(&page->count, 1);
    
    // 3. 加入哈希表和 LRU
    hlist_add_head(&page->hash, head);
    list_add_tail(&page->lru, &page_cache_lru);
    
    spin_unlock(&page_cache_lock);
    return page;
}
```

### 2.3 页释放与回收

```c
void put_page(struct page_cache_entry *page) {
    if (atomic_dec_and_test(&page->count)) {
        spin_lock(&page_cache_lock);
        hlist_del(&page->hash);
        list_del(&page->lru);
        spin_unlock(&page_cache_lock);
        
        kfree(page->data);
        kfree(page);
    }
}
```

---

## 第三章：Buffered I/O 路径实现

### 3.1 write() 系统调用

```c
// fs/file.c
ssize_t sys_write(int fd, const void *buf, size_t count) {
    struct vfs_file *file = get_file(fd);
    if (!file || !file->f_op || !file->f_op->write_iter) {
        return -1;
    }
    
    // 1. 验证用户缓冲区
    if (!validate_user_ptr(buf, count)) {
        return -1;
    }
    
    // 2. 创建 iovec（简化：单缓冲区）
    struct iov_iter iter;
    iov_iter_init(&iter, WRITE, buf, count);
    
    // 3. 调用文件系统 write_iter
    ssize_t ret = file->f_op->write_iter(file, &iter);
    if (ret > 0) {
        file->f_pos += ret;
    }
    return ret;
}
```

### 3.2 通用 write_iter 实现

```c
// fs/generic_file.c
ssize_t generic_file_write_iter(struct vfs_file *file, struct iov_iter *iter) {
    struct vfs_inode *inode = file->f_inode;
    loff_t pos = file->f_pos;
    size_t count = iter->count;
    
    // 1. 循环处理每个页
    while (count > 0) {
        uint64_t index = pos >> PAGE_CACHE_SHIFT;
        uint32_t offset = pos & (PAGE_CACHE_SIZE - 1);
        uint32_t bytes = min(count, PAGE_CACHE_SIZE - offset);
        
        // 2. 获取页
        struct page_cache_entry *page = find_or_create_page(inode, index);
        if (!page) return -1;
        
        // 3. 复制数据到页
        if (copy_from_user((char*)page->data + offset, iter->iov_base, bytes)) {
            put_page(page);
            return -1;
        }
        
        // 4. 标记页为 dirty 和 uptodate
        page->flags |= PG_DIRTY | PG_UPTODATE;
        
        put_page(page);
        
        // 5. 更新位置
        pos += bytes;
        count -= bytes;
        iter->iov_base += bytes;
    }
    
    // 6. 更新文件大小
    if (pos > inode->i_size) {
        inode->i_size = pos;
    }
    
    return iter->count;
}
```

---

## 第四章：回写机制（Writeback）

### 4.1 脏页管理

#### 脏页标记：
- **PG_DIRTY**：页已修改，需写回磁盘
- **回写触发条件**：
  - **内存压力**：Page Cache 满
  - **定时回写**：定期 flush
  - **显式同步**：`fsync()`、`sync()`

### 4.2 回写线程（简化版）

```c
// mm/writeback.c
static void writeback_thread(void *arg) {
    while (1) {
        // 1. 睡眠 5 秒
        sleep(5000);
        
        // 2. 回写脏页
        writeback_dirty_pages();
    }
}

static void writeback_dirty_pages(void) {
    struct page_cache_entry *page, *tmp;
    
    spin_lock(&page_cache_lock);
    list_for_each_entry_safe(page, tmp, &page_cache_lru, lru) {
        if (page->flags & PG_DIRTY) {
            // 1. 清除 dirty 标志（避免重复回写）
            page->flags &= ~PG_DIRTY;
            spin_unlock(&page_cache_lock);
            
            // 2. 写入磁盘
            write_page_to_disk(page);
            
            spin_lock(&page_cache_lock);
        }
    }
    spin_unlock(&page_cache_lock);
}
```

### 4.3 显式同步

```c
// sys_vfs.c
int sys_fsync(int fd) {
    struct vfs_file *file = get_file(fd);
    if (!file) return -1;
    
    // 1. 回写文件所有脏页
    struct vfs_inode *inode = file->f_inode;
    writeback_inode_pages(inode);
    
    // 2. 同步 inode 元数据
    if (inode->i_sb->s_op->write_inode) {
        inode->i_sb->s_op->write_inode(inode, NULL);
    }
    
    return 0;
}
```

---

## 第五章：Direct I/O 支持

### 5.1 Direct I/O 检测

```c
// fs/generic_file.c
ssize_t generic_file_write_iter(struct vfs_file *file, struct iov_iter *iter) {
    // 检查是否 Direct I/O
    if (file->f_flags & O_DIRECT) {
        return generic_file_direct_write(file, iter);
    }
    
    // Buffered I/O 路径
    return generic_file_buffered_write(file, iter);
}
```

### 5.2 Direct I/O 实现

```c
static ssize_t generic_file_direct_write(struct vfs_file *file, 
                                        struct iov_iter *iter) {
    struct vfs_inode *inode = file->f_inode;
    loff_t pos = file->f_pos;
    size_t count = iter->count;
    
    // 1. 对齐检查（简化：要求 512 字节对齐）
    if ((pos & 511) || (count & 511)) {
        return -1;
    }
    
    // 2. 直接写入块设备
    uint64_t block = pos / 512;
    uint32_t blocks = count / 512;
    
    if (block_write(inode->i_sb->s_bdev, block, iter->iov_base, blocks) < 0) {
        return -1;
    }
    
    // 3. 更新文件大小
    if (pos + count > inode->i_size) {
        inode->i_size = pos + count;
    }
    
    return count;
}
```

---

## 第六章：I/O 调度器

### 6.1 请求队列设计

```c
// block/blk-core.h
struct request {
    struct list_head queuelist;
    uint64_t sector;
    void *buffer;
    uint32_t count;
    bool write;
    int result;
    struct block_device *bdev;
};

struct request_queue {
    struct list_head queue;
    spinlock_t lock;
    void (*request_fn)(struct request_queue *q);
};
```

### 6.2 noop 调度器（FIFO）

```c
// block/noop-iosched.c
static void noop_request_fn(struct request_queue *q) {
    struct request *req;
    
    spin_lock(&q->lock);
    if (!list_empty(&q->queue)) {
        req = list_first_entry(&q->queue, struct request, queuelist);
        list_del(&req->queuelist);
    } else {
        req = NULL;
    }
    spin_unlock(&q->lock);
    
    if (req) {
        // 提交到驱动
        req->bdev->ops->submit_request(req->bdev, 
                                       req->sector, 
                                       req->buffer, 
                                       req->count, 
                                       req->write);
        req->result = 0;
        // 通知完成（简化：同步）
        kfree(req);
    }
}
```

### 6.3 请求提交

```c
// block/blk-core.c
int submit_bio(struct bio *bio) {
    struct block_device *bdev = bio->bi_bdev;
    struct request_queue *q = bdev->queue;
    
    // 1. 创建请求
    struct request *req = kmalloc(sizeof(struct request));
    req->sector = bio->bi_sector;
    req->buffer = bio->bi_io_vec[0].bv_page;
    req->count = bio->bi_size / 512;
    req->write = (bio->bi_opf & REQ_OP_WRITE);
    req->bdev = bdev;
    
    // 2. 加入队列
    spin_lock(&q->lock);
    list_add_tail(&req->queuelist, &q->queue);
    spin_unlock(&q->lock);
    
    // 3. 触发调度
    q->request_fn(q);
    return 0;
}
```

---

## 第七章：异步 I/O（AIO）与 io_uring

### 7.1 异步 I/O 基础

#### 问题：
- **同步 I/O 阻塞线程**：write() 返回前不能做其他事
- **多线程开销大**：每个 I/O 一个线程

#### 解决方案：
- **异步 I/O**：提交 I/O 后立即返回，完成后通知

### 7.2 简化 AIO 实现

```c
// aio/aio.c
struct aio_context {
    uint32_t id;
    struct list_head requests;
    struct list_head completed;
    spinlock_t lock;
};

struct aio_request {
    struct list_head list;
    uint32_t aio_id;
    int fd;
    void *buf;
    size_t count;
    off_t offset;
    int result;
    bool completed;
};

// 提交 AIO 请求
int sys_io_submit(aio_context_t ctx_id, long nr, struct iocb **iocbpp) {
    struct aio_context *ctx = get_aio_context(ctx_id);
    for (int i = 0; i < nr; i++) {
        struct iocb *iocb = iocbpp[i];
        struct aio_request *req = kmalloc(sizeof(struct aio_request));
        
        req->aio_id = iocb->aio_id;
        req->fd = iocb->aio_fildes;
        req->buf = (void*)iocb->aio_buf;
        req->count = iocb->aio_nbytes;
        req->offset = iocb->aio_offset;
        req->completed = false;
        
        // 1. 加入请求队列
        spin_lock(&ctx->lock);
        list_add_tail(&req->list, &ctx->requests);
        spin_unlock(&ctx->lock);
        
        // 2. 提交到 I/O 线程（简化：立即执行）
        aio_execute_request(req);
    }
    return nr;
}

// 执行 AIO 请求（简化：同步执行）
static void aio_execute_request(struct aio_request *req) {
    // 执行 write
    int fd = req->fd;
    struct vfs_file *file = get_file(fd);
    // ... 执行 buffered/direct write
    
    req->result = actual_bytes_written;
    req->completed = true;
    
    // 加入完成队列
    struct aio_context *ctx = get_aio_context_by_req(req);
    spin_lock(&ctx->lock);
    list_add_tail(&req->list, &ctx->completed);
    spin_unlock(&ctx->lock);
}
```

### 7.3 io_uring 简化思想

#### io_uring 核心优势：
- **无锁提交/完成**：通过内存环形缓冲区
- **高效批量处理**：一次系统调用处理多个 I/O

#### 环形缓冲区设计：
```c
// aio/io_uring.h
struct io_uring_sq {
    uint32_t *head, *tail, *ring_mask, *ring_entries;
    uint32_t *array; // 请求索引数组
    struct io_uring_sqe *sqes; // 提交队列项
};

struct io_uring_cq {
    uint32_t *head, *tail, *ring_mask, *ring_entries;
    struct io_uring_cqe *cqes; // 完成队列项
};
```

> 💡 **自制 OS 初期可忽略 io_uring，但需理解其无锁思想**

---

## 第八章：I/O 路径整合

### 8.1 完整 I/O 路径

```
+------------------+
|   用户程序        |  // write(fd, buf, 4096)
+------------------+
|   系统调用层      |  // sys_write → iov_iter
+------------------+
|   VFS 层         |  // file->f_op->write_iter
+------------------+
|   Page Cache     |  // find_or_create_page → copy_to_page
+------------------+
|   文件系统层      |  // ext2_write_begin/end
+------------------+
|   块设备层        |  // submit_bio → request_queue
+------------------+
|   I/O 调度器      |  // noop_request_fn
+------------------+
|   驱动层         |  // ide_submit_request
+------------------+
|   硬件           |  // 磁盘
+------------------+
```

### 8.2 回写触发点

| 触发条件 | 行为 |
|----------|------|
| **内存压力** | 回写 LRU 队列尾部的脏页 |
| **定时器** | 每 5 秒回写部分脏页 |
| **fsync()** | 回写文件所有脏页 + 元数据 |
| **文件关闭** | 回写文件脏页 |

---

## 结论：构建可靠的 I/O 栈

I/O 路径是自制操作系统**性能与可靠性的核心**。  
通过 Page Cache、回写机制、I/O 调度器的协同，  
我们实现了：
- **高性能**：缓存减少磁盘 I/O
- **可靠性**：可控的回写策略
- **灵活性**：支持 Buffered/Direct I/O
- **可扩展**：预留 AIO 接口

此 I/O 框架为后续实现 **数据库、高性能服务器** 奠定了坚实基础。  
真正的存储系统，始于对 I/O 路径的深刻理解。

---

## 附录：关键接口速查

### Page Cache
```c
struct page_cache_entry *find_or_create_page(struct vfs_inode *inode, uint64_t index);
void put_page(struct page_cache_entry *page);
```

### I/O 提交
```c
int submit_bio(struct bio *bio);
```

### 系统调用
```c
ssize_t sys_write(int fd, const void *buf, size_t count);
int sys_fsync(int fd);
int sys_io_submit(aio_context_t ctx_id, long nr, struct iocb **iocbpp);
```

### I/O 调度器
```c
struct request_queue *blk_alloc_queue(request_fn *fn);
void blk_queue_bio(struct request_queue *q, struct bio *bio);
```

> **注**：本文所有代码均为简化实现，实际使用需添加错误处理、并发控制、内存管理等。