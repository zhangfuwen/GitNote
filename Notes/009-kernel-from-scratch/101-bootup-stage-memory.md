# 内存管理-启动篇：从实模式到分页开启的内存布局

> **“内核代码链接在 1MB 以上，但启动时 CPU 还在实模式——  
> 今天，我们揭秘启动阶段的内存布局，看内核如何安全开启分页！”**

在操作系统的启动过程中，**内存管理是最先面临的挑战**。  
CPU 上电后处于 **实模式（Real Mode）**，只能访问 1MB 以下内存，  
但内核代码却链接在 **1MB 以上**（如 `0x100000`）。  

更复杂的是：**分页（Paging）开启前，所有地址都是物理地址**；  
开启后，地址变为虚拟地址，硬件自动翻译。  

今天，我们就来：  
✅ **解析启动阶段的内存布局**  
✅ **理解临时页表（Identity Mapping）的作用**  
✅ **掌握安全开启分页的完整流程**  
✅ **获取物理内存大小信息**  

让你彻底搞懂：**内核如何从实模式安全过渡到保护模式 + 分页**！

---

## 🗺️ 一、启动阶段内存布局全景图

### 内存区域划分（0x00000000 - 0xFFFFFFFF）：
```
0xFFFFFFFF +------------------+
           |   内核高半映射   | ← 分页开启后
           +------------------+
0xC0000000 +------------------+
           |      空洞        |
           +------------------+
0x00100000 +------------------+ ← 内核加载地址（1MB）
           |   内核代码/数据  |
           +------------------+
0x0009FC00 +------------------+ ← MP Floating Pointer（多核）
           |   BIOS 数据区    |
           +------------------+
0x00008000 +------------------+ ← GRUB Multiboot 信息
           |   内核 Multiboot 头 |
           +------------------+
0x00007C00 +------------------+ ← 引导扇区（若自写 bootloader）
           |   BIOS 中断向量表 |
           +------------------+
0x00000000 +------------------+
```

### 关键区域说明：
| 地址 | 用途 | 所有者 |
|------|------|--------|
| **0x00000000 - 0x00000400** | 中断向量表 | BIOS |
| **0x00007C00 - 0x00007DFF** | 引导扇区（512B）| Bootloader |
| **0x00008000+** | GRUB 传递的 Multiboot 信息 | GRUB |
| **0x0009FC00+** | MP 表（多核 CPU 信息）| BIOS |
| **0x00100000+** | **内核代码/数据** | 你的内核 |
| **1MB 以下** | **BIOS 保留区** | BIOS |

> 💡 **内核绝不能使用 1MB 以下内存**！否则会覆盖 BIOS 数据，导致崩溃。

---

## 🚪 二、为什么内核要链接在 1MB 以上？

### 历史原因：
- **实模式限制**：20 位地址线 → 最大 1MB（0x00000 - 0xFFFFF）
- **保护模式需求**：32 位 CPU 需要更大地址空间
- **GRUB 约定**：Multiboot 规范要求内核加载到 1MB 以上

### 技术优势：
- **避免与 BIOS 冲突**：1MB 以下是 BIOS 的“领地”
- **简化内存管理**：1MB 以上可视为“干净”的物理内存
- **对齐分页**：1MB = 256 个 4KB 页，便于页表对齐

> ✅ **`0x100000`（1MB）是 x86 内核的“安全起点”**！

---

## 🔗 三、链接脚本：控制内核内存布局

内核的内存布局由 **链接脚本（linker.ld）** 决定：

```ld
ENTRY(_start)

SECTIONS {
    /* 从 1MB 开始 */
    . = 0x100000;

    .text : {
        *(.multiboot)   /* Multiboot 头（必须在前 8KB）*/
        *(.text)        /* 代码段 */
    }

    .rodata : {         /* 只读数据 */
        *(.rodata)
    }

    .data : {           /* 已初始化数据 */
        *(.data)
    }

    .bss : {            /* 未初始化数据（清零）*/
        *(.bss)
        *(COMMON)
    }
}
```

### 关键点：
- **`. = 0x100000`**：设置加载地址（VMA）为 1MB
- **`.multiboot` 段**：必须位于内核前 8KB，GRUB 会扫描它
- **`.bss` 段**：不占 ELF 文件空间，但链接器分配内存并清零

> ⚠️ **若链接地址错误（如 0x0），内核会覆盖中断向量表，立即崩溃**！

---

## 🧱 四、临时页表：Identity Mapping

分页开启前，CPU 访问的是**物理地址**；  
开启后，访问的是**虚拟地址**，需通过页表翻译。  

但内核代码已链接在高地址（如 `0xC0000000`），  
如何在开启分页**瞬间**不崩溃？  

答案：**临时页表（Identity Mapping）**！

### 什么是 Identity Mapping？
- **虚拟地址 = 物理地址**  
  例如：虚拟 `0x100000` → 物理 `0x100000`
- **仅用于启动阶段**，后续会被高半内核映射替换

### 临时页表结构：
```c
// 页目录（1 个，4KB）
uint32_t page_directory[1024] __attribute__((aligned(4096)));

// 页表（1 个，映射前 4MB）
uint32_t first_page_table[1024] __attribute__((aligned(4096)));
```

### 初始化页表：
```nasm
; 汇编中初始化（因 C 未就绪）
setup_paging:
    ; 1. 清零页目录/页表
    mov edi, page_directory
    mov ecx, 1024
    mov eax, 0
    rep stosd

    mov edi, first_page_table
    mov ecx, 1024
    rep stosd

    ; 2. 填充页表：映射 0x00000000 - 0x003FFFFF
    mov ebx, 0
    mov edi, first_page_table
    mov ecx, 1024
.fill_pt:
    mov eax, ebx
    or eax, 0x3  ; PRESENT + RW
    mov [edi], eax
    add ebx, 4096
    add edi, 4
    loop .fill_pt

    ; 3. 页目录指向页表
    mov eax, first_page_table
    or eax, 0x3
    mov [page_directory], eax  ; PDE[0] = 页表地址

    ret
```

> 🔑 **页表项标志 `0x3` = PRESENT(1) + RW(1<<1)**

---

## ⚙️ 五、安全开启分页的完整流程

### 步骤 1：关闭中断
```nasm
cli    ; 禁用可屏蔽中断
```

### 步骤 2：加载临时页表
```nasm
call setup_paging

; 加载 CR3（页目录基地址）
mov eax, page_directory
mov cr3, eax
```

### 步骤 3：开启分页
```nasm
; 设置 CR0.PG = 1
mov eax, cr0
or eax, 0x80000000  ; bit 31 = PG
mov cr0, eax
```

### 步骤 4：刷新流水线（关键！）
```nasm
; 远跳转刷新 CS，确保 CPU 使用新地址翻译
jmp 0x08:protected_mode_32bit

[bits 32]
protected_mode_32bit:
    ; 现在已在分页模式！
    ; 可安全访问高地址（如 0xC0000000）
```

> ✅ **此流程确保：开启分页后，CPU 能正确翻译当前指令地址**！

---

## 📊 六、获取物理内存大小

GRUB 通过 **Multiboot 信息结构** 传递内存信息：

```c
struct multiboot_info {
    uint32_t flags;
    uint32_t mem_lower;  // 低 1MB 可用内存（KB）
    uint32_t mem_upper;  // 高 1MB 可用内存（KB）
    // ... 其他字段
};

void init_memory_info(uint32_t mb_info) {
    struct multiboot_info *mbi = (void*)mb_info;
    if (mbi->flags & 0x01) {
        total_memory_kb = mbi->mem_lower + mbi->mem_upper;
        printk("Total memory: %u KB\n", total_memory_kb);
    }
}
```

> 💡 **`mem_upper` 是 Buddy 分配器初始化的关键输入**！

---

## ⚠️ 七、常见陷阱与避坑指南

1. **页表未对齐**  
   - 页目录/页表必须 **4KB 对齐**  
   - 使用 `__attribute__((aligned(4096)))`

2. **开启分页后立即访问高地址**  
   - 临时页表仅映射低 4MB，访问 `0xC0000000` 会 Page Fault  
   - **解决方案**：先完成 Buddy 初始化，再建立高半内核映射

3. **未刷新 TLB**  
   - 修改页表后需 `invlpg` 或重新加载 CR3  
   - 启动阶段可忽略（TLB 为空）

4. **Multiboot 信息地址未验证**  
   - 检查 `magic == 0x2BADB002` 再使用 `mb_info`

---

## 💬 写在最后

启动阶段的内存管理，  
是操作系统安全的**第一道防线**。  
临时页表虽简单，  
却是后续所有内存操作的基石。

今天你构建的 Identity Mapping，  
正是 Linux `head_32.S` 中 `initial_page_table` 的简化版。

> 🌟 **好的开始，是成功的一半——尤其在内存管理中**。

---

📬 **动手挑战**：  
修改链接脚本，将内核链接到 `0x200000`（2MB），并调整临时页表映射范围。  
欢迎在评论区分享你的启动日志！

👇 下一篇你想看：**Buddy 分配器深度解析**，还是 **Slab 分配器实现**？

---

**#操作系统 #内核开发 #内存管理 #分页 #启动 #x86 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“memory-boot”**，获取：
> - 完整临时页表初始化代码（汇编 + C）
> - 内存布局图（PNG）
> - Multiboot 信息解析模板