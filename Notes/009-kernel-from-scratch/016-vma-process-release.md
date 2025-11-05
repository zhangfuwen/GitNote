
# 从零写 OS 内核-第十六篇：VMA 与进程退出 —— 安全回收每一片内存

> **“分配内存只是开始，安全释放才是终点。  
> 今天，我们实现虚拟内存区域（VMA）管理，并确保进程退出时滴水不漏地回收所有资源！”**

在前几篇中，我们实现了 Buddy 系统、Slab 分配器、用户态 malloc，  
但进程的虚拟地址空间仍是一团乱麻：  
- 无法跟踪哪些区域已分配  
- 进程退出时，物理页、文件描述符、内核对象可能泄漏  

真正的操作系统，必须能**精确管理每个进程的虚拟内存布局**，  
并在进程结束时**彻底回收所有资源**，避免内存泄漏和资源耗尽。

今天，我们就来实现：  
✅ **虚拟内存区域（VMA）链表**  
✅ **进程退出时的完整资源回收**  
✅ **僵尸进程与 wait 机制完善**

让你的 OS 拥有**健壮的生命周期管理**！

---

## 🗺️ 一、为什么需要 VMA（Virtual Memory Area）？

进程的虚拟地址空间不是一整块，而是由多个**独立区域**组成：

| 区域 | 说明 |
|------|------|
| **代码段** | ELF 的 `.text` 段（只读、可执行）|
| **数据段** | `.data` + `.bss`（可读写）|
| **堆（Heap）** | `brk` 扩展的区域 |
| **内存映射区** | `mmap` 创建的区域（文件/匿名）|
| **栈（Stack）** | 用户栈（通常在高地址）|

### 没有 VMA 的问题：
- **无法知道某虚拟地址属于哪个区域**（Page Fault 处理困难）
- **无法正确释放 `mmap` 区域**
- **无法实现 `munmap` 系统调用**

> 💡 **VMA 是进程虚拟地址空间的“地图”**！

---

## 📦 二、VMA 数据结构设计

每个 VMA 描述一个连续的虚拟内存区域：

```c
#define VM_READ  0x00000001
#define VM_WRITE 0x00000002
#define VM_EXEC  0x00000004
#define VM_SHARED 0x00000008
#define VM_ANONYMOUS 0x00000010
#define VM_FILE  0x00000020

struct vm_area_struct {
    uint32_t vm_start;      // 虚拟地址起始
    uint32_t vm_end;        // 虚拟地址结束
    uint32_t vm_flags;      // 权限与类型标志
    struct file *vm_file;   // 如果是文件映射
    off_t vm_pgoff;         // 文件偏移（页对齐）
    
    struct vm_area_struct *vm_next; // 按地址排序的链表
};
```

### 进程 PCB 增加 VMA 链表：
```c
typedef struct task {
    // ... 其他字段
    struct vm_area_struct *mm; // VMA 链表头
    uint32_t heap_start;    // 堆起始地址
    uint32_t heap_top;      // 当前堆顶
} task_t;
```

> 🔑 **VMA 链表按 `vm_start` 升序排列**，便于查找和合并。

---

## ⚙️ 三、VMA 核心操作

### 1. **查找 VMA（find_vma）**
```c
struct vm_area_struct *find_vma(task_t *task, uint32_t addr) {
    struct vm_area_struct *vma = task->mm;
    while (vma) {
        if (addr >= vma->vm_start && addr < vma->vm_end) {
            return vma;
        }
        vma = vma->vm_next;
    }
    return NULL;
}
```

### 2. **插入 VMA（insert_vma）**
```c
void insert_vma(task_t *task, struct vm_area_struct *new_vma) {
    struct vm_area_struct **link = &task->mm;
    
    // 找到插入位置（保持有序）
    while (*link && (*link)->vm_start < new_vma->vm_start) {
        link = &(*link)->vm_next;
    }
    
    new_vma->vm_next = *link;
    *link = new_vma;
}
```

### 3. **合并相邻 VMA（可选优化）**
```c
void merge_vmas(task_t *task) {
    struct vm_area_struct *vma = task->mm;
    while (vma && vma->vm_next) {
        if (vma->vm_end == vma->vm_next->vm_start &&
            vma->vm_flags == vma->vm_next->vm_flags) {
            // 合并
            vma->vm_end = vma->vm_next->vm_end;
            struct vm_area_struct *next = vma->vm_next;
            vma->vm_next = next->vm_next;
            kfree(next);
        } else {
            vma = vma->vm_next;
        }
    }
}
```

---

## 📞 四、系统调用集成

### 1. **mmap：创建新 VMA**
```c
void *sys_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    // 1. 对齐地址和长度
    uint32_t vaddr = (uint32_t)addr;
    if (!vaddr) {
        vaddr = find_free_area(current_task, length);
    }
    length = ALIGN_UP(length, PAGE_SIZE);
    
    // 2. 创建 VMA
    struct vm_area_struct *vma = kmalloc(sizeof(struct vm_area_struct));
    vma->vm_start = vaddr;
    vma->vm_end = vaddr + length;
    vma->vm_flags = (prot & PROT_READ ? VM_READ : 0) |
                    (prot & PROT_WRITE ? VM_WRITE : 0) |
                    (flags & MAP_ANONYMOUS ? VM_ANONYMOUS : VM_FILE);
    vma->vm_file = (flags & MAP_ANONYMOUS) ? NULL : get_file(fd);
    vma->vm_pgoff = offset / PAGE_SIZE;
    
    // 3. 插入 VMA 链表
    insert_vma(current_task, vma);
    
    // 4. （按需映射物理页，或推迟到 Page Fault）
    return (void*)vaddr;
}
```

### 2. **munmap：释放 VMA**
```c
int sys_munmap(void *addr, size_t length) {
    uint32_t start = (uint32_t)addr;
    uint32_t end = start + ALIGN_UP(length, PAGE_SIZE);
    
    struct vm_area_struct *vma = current_task->mm;
    struct vm_area_struct **prev = &current_task->mm;
    
    while (vma) {
        if (vma->vm_start >= end) break;
        if (vma->vm_end <= start) {
            prev = &vma->vm_next;
            vma = vma->vm_next;
            continue;
        }
        
        // 释放该 VMA 覆盖的物理页
        unmap_vma_pages(vma, start, end);
        
        // 从链表移除
        *prev = vma->vm_next;
        kfree(vma);
        vma = *prev;
    }
    
    return 0;
}
```

---

## ☠️ 五、进程退出：资源回收全景

当进程调用 `exit()` 或被杀死，内核必须回收：

### 1. **虚拟内存资源**
```c
void free_process_memory(task_t *task) {
    // 1. 释放所有 VMA
    struct vm_area_struct *vma = task->mm;
    while (vma) {
        // 释放物理页
        for (uint32_t addr = vma->vm_start; addr < vma->vm_end; addr += PAGE_SIZE) {
            uint32_t paddr = get_phys_addr(task->cr3, addr);
            if (paddr) {
                // 释放物理页（Buddy 系统）
                buddy_free((void*)paddr, 0);
                // 更新页表
                unmap_page(task->cr3, addr);
            }
        }
        // 释放文件引用（如果是文件映射）
        if (vma->vm_file) {
            fput(vma->vm_file);
        }
        struct vm_area_struct *next = vma->vm_next;
        kfree(vma);
        vma = next;
    }
    
    // 2. 释放页目录
    buddy_free((void*)task->cr3, get_order(PAGE_SIZE));
}
```

### 2. **文件描述符**
```c
void free_process_files(task_t *task) {
    for (int i = 0; i < MAX_FDS; i++) {
        if (task->fd_table[i]) {
            close_fd(task->fd_table[i]); // 递减引用计数
            task->fd_table[i] = NULL;
        }
    }
}
```

### 3. **内核对象**
```c
void free_process_kernel_objects(task_t *task) {
    // 释放 PCB 本身（通过 Slab）
    kmem_cache_free(task_cache, task);
}
```

### 4. **进程状态转换**
```c
void do_exit(int exit_code) {
    task_t *task = current_task;
    
    // 1. 释放所有资源
    free_process_memory(task);
    free_process_files(task);
    
    // 2. 通知父进程
    send_sigchild(task->parent, task->pid, exit_code);
    
    // 3. 进入 ZOMBIE 状态（保留 PCB 直到父进程 wait）
    task->state = TASK_ZOMBIE;
    task->exit_code = exit_code;
    
    // 4. 切换到新进程
    schedule();
}
```

---

## 🧪 六、测试：内存泄漏检测

### 用户程序：
```c
void test_leak() {
    // 分配内存
    void *p1 = malloc(1024);
    void *p2 = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_ANONYMOUS, -1, 0);
    
    // 不 free/munmap，直接 exit
    exit(0);
}
```

### 内核监控：
- 在 `do_exit` 中打印释放的 VMA 数量、物理页数
- 确保每次进程退出，资源计数归零

> ✅ **无泄漏：进程退出后，所有物理页、VMA、文件描述符均被回收！**

---

## ⚠️ 七、边界情况处理

1. **多线程进程**  
   - 当前设计为单线程，多线程需额外管理线程栈

2. **共享内存**  
   - `MAP_SHARED` 映射的区域，释放时需检查是否还有其他进程引用

3. **信号处理中的退出**  
   - 确保在 Page Fault、系统调用中也能安全退出

4. **父进程先退出**  
   - 子进程成为孤儿，应被 init 进程收养

> 💡 **完善的进程退出机制，是系统长期稳定运行的关键！**

---

## 💬 写在最后

VMA 和进程退出看似是“收尾工作”，  
但它们决定了操作系统的**健壮性与可靠性**。  
一个会泄漏内存的 OS，终将在长时间运行后崩溃。

今天你实现的 `do_exit`，  
正是 Linux 中 `do_exit` 和 `release_task` 的简化版。

> 🌟 **真正的工程之美，在于优雅地处理每一个结束。**

---

📬 **动手挑战**：  
实现 `munmap` 系统调用，并验证释放后该区域触发 Page Fault。  
欢迎在评论区分享你的 VMA 链表调试技巧！

👇 下一篇你想看：**写时复制（CoW）实现**，还是 **进程间通信（IPC）**？

---

**#操作系统 #内核开发 #VMA #进程退出 #内存管理 #资源回收 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“vma”**，获取：
> - 完整 VMA 管理代码（find/insert/unmap）
> - 进程退出资源回收全流程
> - mmap/munmap 系统调用实现模板
