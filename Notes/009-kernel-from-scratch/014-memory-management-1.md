
# 从零写 OS 内核-第十四篇：内存管理 —— 从页表到 malloc

> **“内存是操作系统的血液，管理它，就是管理生命本身。  
> 今天，我们构建完整的内存管理体系：进程页表、物理页分配、用户态 malloc！”**

在前面的篇章中，我们实现了分页、多进程、文件系统，但内存管理仍很原始：  
- 物理页分配靠简单链表  
- 用户程序无法动态申请内存（没有 `malloc`）  
- 进程页表无法动态扩展  

真正的操作系统，必须能**高效管理物理内存**，并为**用户程序提供虚拟内存服务**。  

今天，我们就来实现：  
✅ **每个进程独立的页表管理**  
✅ **Buddy 系统分配物理页**  
✅ **用户态内存分配器（malloc/free）**，底层通过 `mmap` 系统调用  

让你的用户程序能像在 Linux 中一样自由使用 `malloc`！

---

## 🧱 一、内存管理全景图

现代操作系统内存管理分为三层：

| 层级 | 职责 | 本篇实现 |
|------|------|----------|
| **物理层** | 管理物理内存页（4KB） | ✅ Buddy 分配器 |
| **虚拟层** | 管理进程虚拟地址空间 | ✅ 页表动态映射 |
| **用户层** | 提供 `malloc`/`free` | ✅ 用户态分配器 + `mmap` |

> 💡 **核心思想**：**物理内存 → 虚拟地址 → 用户指针**

---

## 📦 二、Buddy 分配器：高效管理物理页

Buddy 系统将内存划分为 **2^n 页大小的块**，支持快速分配/合并，减少碎片。

### 设计：
- **位图（Bitmap）**：跟踪每个块是否空闲
- **空闲链表数组**：`free_list[0..10]`，`free_list[i]` 存放 2^i 页大小的块

### 关键操作：
#### 1. **初始化**
```c
#define MAX_ORDER 10  // 最大 2^10 = 1024 页 = 4MB
struct list_head free_list[MAX_ORDER];
uint8_t *buddy_bitmap;

void buddy_init(uint32_t start_addr, uint32_t end_addr) {
    uint32_t total_pages = (end_addr - start_addr) / PAGE_SIZE;
    buddy_bitmap = (uint8_t*)kmalloc(BITS_TO_BYTES(total_pages));
    // 将整个内存区域加入 free_list[MAX_ORDER]
    buddy_add_to_free_list(start_addr, MAX_ORDER);
}
```

#### 2. **分配（buddy_alloc）**
```c
void *buddy_alloc(int order) {
    // 1. 从 free_list[order] 取块
    if (!list_empty(&free_list[order])) {
        return remove_from_list(&free_list[order]);
    }
    // 2. 向上找更大块
    for (int i = order + 1; i <= MAX_ORDER; i++) {
        if (!list_empty(&free_list[i])) {
            void *block = remove_from_list(&free_list[i]);
            // 3. 分割：将大块拆成两个 buddy
            buddy_split(block, i, order);
            return block;
        }
    }
    return NULL; // 内存不足
}
```

#### 3. **释放（buddy_free）**
```c
void buddy_free(void *addr, int order) {
    // 1. 标记为 free
    mark_free(addr, order);
    // 2. 尝试合并 buddy
    void *buddy = get_buddy(addr, order);
    if (is_free(buddy, order)) {
        buddy_free(merge(addr, buddy), order + 1);
    } else {
        add_to_free_list(addr, order);
    }
}
```

> ✅ **Buddy 系统保证：分配 N 页，最多浪费 N 页（内部碎片）**

---

## 🗺️ 三、进程页表管理：动态映射虚拟地址

每个进程需要**独立的页目录（Page Directory）**，并支持**按需映射**。

### 1. **页表操作 API**
```c
// 映射虚拟地址到物理页
int map_page(uint32_t *pgdir, uint32_t vaddr, uint32_t paddr, uint32_t flags);

// 分配新页表（如果需要）
uint32_t *get_page_table(uint32_t *pgdir, uint32_t vaddr, bool create);

// 取消映射
void unmap_page(uint32_t *pgdir, uint32_t vaddr);
```

### 2. **按需分配（Demand Paging）**
当用户访问未映射的虚拟地址：
- 触发 **Page Fault**（#PF）
- 内核分配物理页，映射到该地址
- 恢复执行

```c
void page_fault_handler(registers_t *regs) {
    uint32_t fault_addr;
    asm("mov %%cr2, %0" : "=r"(fault_addr));

    if (is_user_address(fault_addr)) {
        // 分配物理页
        void *page = buddy_alloc(0);
        if (page) {
            // 映射到当前进程页目录
            map_page(current_task->cr3, fault_addr & ~0xFFF, 
                     (uint32_t)page, PAGE_PRESENT | PAGE_RW | PAGE_USER);
            return;
        }
    }
    // 否则：段错误（SIGSEGV）
    kill_process(current_task, SIGSEGV);
}
```

> 🔑 **Page Fault 是虚拟内存的基石**！

---

## 📞 四、系统调用：mmap 与 brk

用户程序通过系统调用申请内存：

### 1. **brk 系统调用**（传统）
- 调整 **堆顶指针（program break）**
- 用于 `malloc` 小内存

```c
// sys_brk: 设置新的堆顶
void *sys_brk(void *new_brk) {
    if (!new_brk) return current_task->heap_start;

    uint32_t old_heap = current_task->heap_top;
    uint32_t new_heap = (uint32_t)new_brk;

    if (new_heap > old_heap) {
        // 映射新页面（按需）
        uint32_t vaddr = old_heap & ~0xFFF;
        while (vaddr < new_heap) {
            map_page(current_task->cr3, vaddr, 0, 
                     PAGE_PRESENT | PAGE_RW | PAGE_USER);
            vaddr += PAGE_SIZE;
        }
    }
    current_task->heap_top = new_heap;
    return (void*)new_heap;
}
```

### 2. **mmap 系统调用**（现代）
- 映射**任意虚拟地址范围**
- 可映射文件、设备、匿名内存

```c
void *sys_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    // 1. 找到未使用的虚拟地址范围
    uint32_t vaddr = find_free_vma(length);
    
    // 2. 分配物理页（匿名映射）
    if (flags & MAP_ANONYMOUS) {
        for (size_t i = 0; i < length; i += PAGE_SIZE) {
            void *page = buddy_alloc(0);
            map_page(current_task->cr3, vaddr + i, (uint32_t)page, 
                     PAGE_PRESENT | PAGE_RW | PAGE_USER);
        }
    }
    // 3. （文件映射暂不实现）
    return (void*)vaddr;
}
```

> 💡 **`malloc` 底层：小内存用 `brk`，大内存用 `mmap`**

---

## 🧵 五、用户态内存分配器（malloc/free）

现在，我们可以在用户空间实现 `malloc`！

### 1. **用户态 libc 封装**
```c
// user_malloc.c
extern void *brk(void *addr);
extern void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);

void *malloc(size_t size) {
    if (size == 0) return NULL;

    if (size < 128 * 1024) {
        // 小内存：用 brk 扩展堆
        static char *heap_start = NULL;
        static char *heap_end = NULL;
        
        if (!heap_start) {
            heap_start = brk(NULL);
            heap_end = heap_start;
        }

        char *new_end = heap_end + size;
        brk(new_end);
        heap_end = new_end;
        return new_end - size;
    } else {
        // 大内存：用 mmap
        return mmap(NULL, size, PROT_READ | PROT_WRITE, 
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }
}

void free(void *ptr) {
    // 简化版：不回收小内存（实际需空闲链表）
    // 大内存：munmap
    // ... 
}
```

### 2. **链接用户程序**
```makefile
# 用户程序编译
user_program: user_main.c user_malloc.c
    i686-elf-gcc -ffreestanding -nostdlib -o $@ $^ -T user.ld
```

---

## 🧪 六、测试：用户程序中的 malloc

**user_main.c**：
```c
int main() {
    // 小内存
    int *arr = (int*)malloc(1024 * sizeof(int));
    for (int i = 0; i < 1024; i++) {
        arr[i] = i;
    }
    printf("Small alloc: arr[100] = %d\n", arr[100]);

    // 大内存
    char *big = (char*)malloc(1024 * 1024); // 1MB
    big[0] = 'A';
    big[1024*1024-1] = 'Z';
    printf("Big alloc: %c ... %c\n", big[0], big[1024*1024-1]);

    free(arr);
    free(big);
    return 0;
}
```

运行效果：
```
Small alloc: arr[100] = 100
Big alloc: A ... Z
```

✅ **用户程序成功使用 malloc/free！**

---

## ⚠️ 七、优化方向

1. **用户态 malloc 优化**：
   - 空闲链表（free list）回收小内存
   - 多级分配器（如 dlmalloc）

2. **写时复制（CoW）**：
   - `fork` 时不复制物理页，只设为只读
   - 写时触发 Page Fault，才复制

3. **内存映射文件**：
   - `mmap` 支持映射 ext2 文件内容
   - 实现 demand paging from disk

4. **交换（Swapping）**：
   - 将不活跃页换出到磁盘

---

## 💬 写在最后

从 Buddy 分配器到 `malloc`，  
你构建的不仅是内存管理代码，  
更是**现代操作系统的核心支柱**。

今天你实现的 `mmap`，  
正是 Linux 中 `glibc malloc` 的底层依赖。

> 🌟 **内存管理的终极目标：让每个进程都拥有“无限”且安全的地址空间。**

---

📬 **动手挑战**：  
修改用户态 malloc，支持 free 小内存块并重复使用。  
欢迎在评论区分享你的分配器设计！

👇 下一篇你想看：**写时复制（CoW）实现**，还是 **内存映射文件（mmap file）**？

---

**#操作系统 #内核开发 #内存管理 #Buddy分配器 #malloc #mmap #页表 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“memory”**，获取：
> - 完整 Buddy 分配器代码
> - 页表操作与 Page Fault 处理
> - 用户态 malloc/free 实现模板
