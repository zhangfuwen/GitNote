
# 从零写 OS 内核-第十九篇：用户与内核地址空间 —— 构建安全的虚拟内存边界

> **“所有进程共享内核代码，但绝不能互相窥探；用户程序自由驰骋，却无法越界半步。  
> 今天，我们设计高半内核映射，实现用户与内核地址空间的完美隔离与高效共享！”**

在前面的篇章中，我们实现了分页、多进程、多核，但所有进程的页目录都**完全独立**——  
包括内核代码、数据、页表本身！这导致：
- **每次进程切换都要刷新 TLB**（性能极差）
- **无法高效共享内核资源**
- **内核无法直接访问用户内存**

真正的操作系统，必须采用 **高半内核（Higher Half Kernel）** 设计：  
✅ **低地址空间（0x00000000 - 0xBFFFFFFF）**：用户空间（每个进程独立）  
✅ **高地址空间（0xC0000000 - 0xFFFFFFFF）**：内核空间（所有进程共享）  

今天，我们就来重构内存布局，实现**安全又高效**的地址空间管理！

---

## 🏗️ 一、为什么需要高半内核？

### 传统独立页目录的问题：
| 问题 | 后果 |
|------|------|
| **TLB 频繁刷新** | 每次进程切换，所有 TLB 条目失效 → 性能下降 30%+ |
| **内核无法访问用户内存** | 系统调用需复杂地址转换 |
| **内核内存浪费** | 每个进程都映射一份内核代码（4GB × N 进程）|

### 高半内核的优势：
- **TLB 友好**：内核映射不变，切换进程时 TLB 无需刷新
- **高效访问**：内核可直接通过高地址访问用户内存（通过临时映射）
- **节省内存**：内核代码/数据只映射一次

> 💡 **Linux、Windows、macOS 全部采用高半内核设计**！

---

## 🗺️ 二、地址空间布局设计

### 32 位 x86 经典布局（4GB 虚拟地址）：
```
0xFFFFFFFF +------------------+
           |   内核栈         | ← 每个进程独立（高地址向下增长）
           +------------------+
           |   内核虚拟内存   | ← vmalloc 区域（动态映射）
           +------------------+
           |   内核直接映射   | ← 物理内存线性映射（0xC0000000 = 0x00000000）
           +------------------+
0xC0000000 +------------------+ ← **内核空间起点**
           |                  |
           |     空洞         | ← 通常 1GB 空洞（安全隔离）
           |                  |
0xBFFFFFFF +------------------+ ← **用户空间终点**
           |   用户栈         | ← 高地址向下增长
           +------------------+
           |   内存映射区     | ← mmap 区域
           +------------------+
           |   堆（Heap）     | ← brk 向上增长
           +------------------+
           |   数据段         |
           +------------------+
           |   代码段         |
0x00000000 +------------------+
```

### 关键常量：
```c
#define KERNEL_VIRTUAL_BASE 0xC0000000  // 内核虚拟基地址
#define USER_VIRTUAL_MAX    0xBFFFFFFF  // 用户空间最大地址
```

> 🔑 **内核直接映射：`virtual_addr = physical_addr + KERNEL_VIRTUAL_BASE`**

---

## 🔧 三、重构页目录：共享内核映射

### 1. **创建内核页目录模板**
```c
// 内核页目录（只包含内核映射）
uint32_t kernel_page_dir[1024] __attribute__((aligned(4096)));

void setup_kernel_page_dir() {
    // 1. 映射内核代码/数据（直接映射）
    for (uint32_t paddr = 0; paddr < kernel_size; paddr += PAGE_SIZE) {
        uint32_t vaddr = paddr + KERNEL_VIRTUAL_BASE;
        map_page(kernel_page_dir, vaddr, paddr, 
                 PAGE_PRESENT | PAGE_RW | PAGE_GLOBAL);
    }
    
    // 2. 映射 LAPIC、IOAPIC 等硬件
    map_page(kernel_page_dir, 0xFEE00000, 0xFEE00000, 
             PAGE_PRESENT | PAGE_RW | PAGE_GLOBAL);
    
    // 3. 递归映射页目录自身（用于页表操作）
    kernel_page_dir[1023] = (uint32_t)kernel_page_dir | 3;
}
```

> 🌟 **`PAGE_GLOBAL` 标志**：此页在进程切换时**不刷新 TLB**（性能关键！）

### 2. **为每个进程创建页目录**
```c
uint32_t *create_user_page_dir() {
    // 1. 分配新页目录
    uint32_t *pgdir = buddy_alloc(0);
    memset(pgdir, 0, PAGE_SIZE);
    
    // 2. 复制内核映射（高 1GB）
    memcpy(&pgdir[768], &kernel_page_dir[768], 256 * sizeof(uint32_t));
    // 768 = 0xC0000000 / (4MB per PDE) → 768-1023 项为内核
    
    // 3. 用户空间留空（按需分配）
    return pgdir;
}
```

> ✅ **每个进程页目录的高 256 项（768-1023）与内核页目录相同**！

---

## 🔄 四、进程切换优化：无需刷新 TLB

### 传统切换（低效）：
```c
void switch_to_task(task_t *next) {
    // 加载新页目录
    asm volatile ("mov %0, %%cr3" :: "r"(next->cr3));
    // → 所有 TLB 条目失效！
}
```

### 高半内核切换（高效）：
```c
void switch_to_task(task_t *next) {
    // 仅当页目录真正改变时才加载 CR3
    if (next->cr3 != current_task->cr3) {
        asm volatile ("mov %0, %%cr3" :: "r"(next->cr3));
        // → 但内核 TLB 条目因 PAGE_GLOBAL 保留！
    }
    current_task = next;
}
```

> 📊 **性能提升**：TLB 未命中率降低 50%+，尤其在频繁系统调用时！

---

## 📞 五、系统调用：安全访问用户内存

内核现在运行在高地址（0xC0000000+），但用户指针是低地址（0x00000000+）。  
如何安全访问？

### 1. **验证用户指针**
```c
bool is_user_pointer(void *ptr) {
    return (uint32_t)ptr < USER_VIRTUAL_MAX;
}
```

### 2. **临时映射（推荐）**
- 使用**固定内核虚拟地址窗口**（如 0xFFC00000）临时映射用户物理页
- 避免直接使用用户虚拟地址（可能触发 Page Fault）

```c
char *temp_user_map(void *user_ptr) {
    uint32_t paddr = get_phys_addr(current_task->cr3, (uint32_t)user_ptr);
    if (!paddr) return NULL;
    
    // 将物理页映射到临时窗口
    map_page(kernel_page_dir, TEMP_WINDOW_VADDR, paddr, 
             PAGE_PRESENT | PAGE_RW);
    invlpg(TEMP_WINDOW_VADDR); // 刷新 TLB
    
    return (char*)(TEMP_WINDOW_VADDR + ((uint32_t)user_ptr & 0xFFF));
}
```

### 3. **系统调用中的使用**
```c
int sys_write(int fd, const void *buf, size_t count) {
    // 1. 验证用户指针
    if (!is_user_pointer((void*)buf)) {
        return -1;
    }
    
    // 2. 临时映射
    char *kernel_buf = temp_user_map((void*)buf);
    if (!kernel_buf) return -1;
    
    // 3. 内核直接操作 kernel_buf
    console_write(kernel_buf, count);
    
    return count;
}
```

> ⚠️ **绝对不要直接解引用用户指针**！可能触发 Page Fault 或访问非法地址。

---

## 🧪 六、测试：地址空间隔离验证

### 用户程序 1：
```c
void test1() {
    char *p = (char*)0xC0000000; // 尝试访问内核空间
    *p = 'A'; // 应触发 Page Fault（段错误）
}
```

### 用户程序 2：
```c
void test2() {
    char *p = (char*)0x100000; // 用户空间
    *p = 'B'; // 正常
}
```

### 内核行为：
- `test1` 触发 **Page Fault** → 内核发送 `SIGSEGV` → 进程终止
- `test2` 正常执行

✅ **用户无法访问内核空间，隔离成功！**

---

## ⚠️ 七、高级话题：递归页表与页表自映射

### 问题：内核如何操作任意进程的页表？
- 页表本身也在虚拟内存中
- 需要一种方式通过虚拟地址访问页表

### 解决方案：**递归页表（Recursive Page Table）**
- 将页目录的最后一项（PDE 1023）指向**页目录自身**
- 形成虚拟地址 → 页目录 → 页表 → 页 的映射

### 虚拟地址布局：
```
0xFFC00000 - 0xFFFFFFFF: 页表自映射区域
  - 0xFFFFF000: 页目录本身
  - 0xFFFFE000: 第 1022 个页表
  - ...
```

### 代码：
```c
#define RECURSIVE_PDE_INDEX 1023
#define RECURSIVE_VADDR_BASE (RECURSIVE_PDE_INDEX * 4 * 1024 * 1024) // 0xFFC00000

void setup_recursive_mapping(uint32_t *pgdir) {
    pgdir[RECURSIVE_PDE_INDEX] = (uint32_t)pgdir | 3;
}

// 通过虚拟地址访问页表
uint32_t *get_page_table_vaddr(uint32_t *pgdir, uint32_t vaddr) {
    uint32_t pd_index = (vaddr >> 22) & 0x3FF;
    uint32_t pt_index = (vaddr >> 12) & 0x3FF;
    return (uint32_t*)(RECURSIVE_VADDR_BASE + 
                      (pd_index * 4096) + 
                      (pt_index * 4));
}
```

> 💡 **递归页表是 Linux 内核操作页表的核心机制**！

---

## 💬 写在最后

用户与内核地址空间的设计，  
是操作系统**安全与性能的平衡艺术**。  
高半内核不仅提升了性能，  
更构建了坚不可摧的内存隔离墙。

今天你设置的 `0xC0000000`，  
正是无数操作系统内核的“安全港湾”。

> 🌟 **最好的隔离，是让用户感觉不到隔离的存在。**

---

📬 **动手挑战**：  
修改你的内核，将内核映射到 `0xC0000000`，并验证用户程序无法访问高地址。  
欢迎在评论区分享你的地址空间布局图！

👇 下一篇你想看：**进程间通信（管道/Pipe）**，还是 **信号（Signal）机制**？

---

**#操作系统 #内核开发 #虚拟内存 #高半内核 #地址空间 #内存隔离 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“address”**，获取：
> - 完整高半内核页目录设置代码
> - 临时映射与用户内存安全访问模板
> - 递归页表实现与调试技巧
