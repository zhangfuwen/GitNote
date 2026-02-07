# 玩转 BIOS：从启动到接管硬件的完整指南

> **"BIOS 不仅是启动代码，更是硬件的终极控制台。  
> 本文将深入 BIOS 的启动流程、中断服务、硬件检测，  
> 为你揭示如何在实模式下完全掌控 x86 硬件。"**

## 引言：BIOS 的现代价值

在 UEFI 时代，许多人认为 BIOS 已经过时。  
但对自制操作系统开发者来说，**BIOS 仍是最佳的学习起点**：

- **简化启动**：无需处理复杂的 UEFI 协议
- **硬件抽象**：提供标准中断服务（INT 10h, INT 13h, INT 16h）
- **调试友好**：实模式调试比保护模式简单得多
- **兼容性好**：所有 x86 PC 都支持 BIOS 启动

本文将带你**玩转 BIOS**，从启动流程到硬件控制，  
为自制 OS 构建坚实的启动基础。

---

## 第一章：BIOS 启动流程深度解析

### 1.1 上电自检（POST）阶段

#### BIOS 启动序列
```
1. 上电 → CPU 复位 → CS:IP = 0xF000:0xFFF0
2. 执行跳转指令 → BIOS 代码段
3. 执行 POST（Power-On Self Test）
4. 检测硬件（内存、显卡、键盘、磁盘）
5. 初始化硬件（设置中断向量表、配置 DMA）
6. 加载启动设备（MBR）
7. 跳转到 MBR（0x7C00）
```

#### 关键内存布局
| 地址范围 | 用途 |
|----------|------|
| **0x0000-0x03FF** | 中断向量表（IVT）|
| **0x0400-0x04FF** | BIOS 数据区（BDA）|
| **0x0500-0x7BFF** | 可用内存 |
| **0x7C00-0x7DFF** | 引导扇区 |
| **0x7E00-0x9FFF** | 引导程序堆栈 |
| **0xA000-0xBFFF** | 显存 |
| **0xC000-0xDFFF** | BIOS 扩展 ROM |
| **0xE000-0xFFFF** | BIOS ROM |

### 1.2 中断向量表（IVT）

#### IVT 结构
```assembly
; 中断向量表（每个中断 4 字节：偏移 + 段）
; 地址 0x0000:0x0000
dw timer_handler    ; INT 08h - 时钟中断
dw 0x0000
dw keyboard_handler ; INT 09h - 键盘中断  
dw 0x0000
dw video_handler    ; INT 10h - 视频服务
dw 0x0000
dw disk_handler     ; INT 13h - 磁盘服务
dw 0x0000
dw serial_handler   ; INT 14h - 串口服务
dw 0x0000
dw mouse_handler    ; INT 33h - 鼠标服务
dw 0x0000
```

#### 自定义中断处理
```assembly
; 安装自定义时钟中断处理程序
install_timer_handler:
    cli                     ; 禁用中断
    mov ax, cs
    mov [0x0020], timer_handler  ; INT 08h 偏移
    mov [0x0022], ax             ; INT 08h 段
    sti                     ; 启用中断
    ret

timer_handler:
    push ax
    push dx
    
    ; 更新系统时间
    inc word [timer_ticks]
    
    ; 调用原始 BIOS 中断（可选）
    pushf
    call far [0x0024]       ; 原始 INT 08h
    
    pop dx
    pop ax
    iret
```

### 1.3 BIOS 数据区（BDA）

#### BDA 关键字段
| 偏移 | 大小 | 用途 |
|------|------|------|
| **0x0410** | word | 设备列表（bit 7=DMA, bit 6=FDD, bit 5=LPT2, bit 4=LPT1, bit 3=RS232, bit 2=Game, bit 1=Tape, bit 0=FDD）|
| **0x0413** | word | 基础内存大小（KB）|
| **0x0449** | byte | 当前视频模式 |
| **0x0450** | word | 当前光标位置（高字节=行，低字节=列）|
| **0x0463** | dword | BIOS 视频服务入口 |

#### 读取 BDA 信息
```c
// 读取基础内存大小
uint16_t get_base_memory(void) {
    uint16_t *bda_base_mem = (uint16_t*)0x0413;
    return *bda_base_mem;
}

// 读取视频模式
uint8_t get_video_mode(void) {
    uint8_t *bda_video_mode = (uint8_t*)0x0449;
    return *bda_video_mode;
}
```

---

## 第二章：BIOS 中断服务详解

### 2.1 视频服务（INT 10h）

#### 常用功能
| AH | 功能 | 参数 |
|----|------|------|
| **0x00** | 设置视频模式 | AL=模式 |
| **0x02** | 设置光标位置 | DH=行, DL=列, BH=页 |
| **0x03** | 读取光标位置 | BH=页 |
| **0x06** | 滚动窗口向上 | AL=行数, BH=属性, CH=左上行, CL=左上列, DH=右下行, DL=右下列 |
| **0x0E** | 电传打字输出 | AL=字符, BH=页, BL=前景色 |

#### 文本模式操作
```assembly
; 设置 80x25 文本模式
mov ah, 0x00
mov al, 0x03
int 0x10

; 设置光标到 (10, 20)
mov ah, 0x02
mov bh, 0x00    ; 页 0
mov dh, 10      ; 行 10
mov dl, 20      ; 列 20
int 0x10

; 输出字符 'A'
mov ah, 0x0E
mov al, 'A'
mov bh, 0x00
mov bl, 0x07    ; 白色前景，黑色背景
int 0x10
```

#### 图形模式操作
```assembly
; 设置 320x200 256 色模式
mov ah, 0x00
mov al, 0x13
int 0x10

; 写像素 (x=100, y=50, color=15)
mov ah, 0x0C
mov al, 15      ; 颜色
mov cx, 100     ; X 坐标
mov dx, 50      ; Y 坐标
int 0x10

; 读像素
mov ah, 0x0D
mov cx, 100
mov dx, 50
int 0x10
; 返回颜色在 AL
```

### 2.2 磁盘服务（INT 13h）

#### 磁盘参数
- **CHS 寻址**：柱面（Cylinder）、磁头（Head）、扇区（Sector）
- **扇区大小**：512 字节
- **最大容量**：8.4GB（1024 柱面 × 256 磁头 × 63 扇区 × 512 字节）

#### 读取扇区
```assembly
; 读取 1 个扇区到 0x7E00
read_sector:
    pusha
    
    mov ah, 0x02        ; 读取扇区功能
    mov al, 0x01        ; 扇区数
    mov ch, 0x00        ; 柱面
    mov cl, 0x02        ; 扇区（1-63）
    mov dh, 0x00        ; 磁头
    mov dl, 0x80        ; 驱动器（0x80=第一硬盘）
    mov bx, 0x7E00      ; 缓冲区地址
    int 0x13
    
    jc disk_error       ; CF=1 表示错误
    
    popa
    ret

disk_error:
    mov si, error_msg
    call print_string
    hlt

error_msg: db "Disk error!", 0
```

#### 磁盘参数表（DPT）
```c
// 获取磁盘参数
struct disk_parameter_table {
    uint16_t buffer;        // 缓冲区偏移
    uint16_t count;         // 扇区数
    uint16_t sector;        // 起始扇区
    uint16_t cylinder;      // 起始柱面
    uint8_t head;           // 起始磁头
    uint8_t drive;          // 驱动器号
    uint8_t flags;          // 标志
};

// 扩展磁盘服务（INT 13h, AH=48h）
int get_disk_info(uint8_t drive, struct disk_parameter_table *dpt) {
    struct {
        uint16_t size;      // 结构大小
        uint16_t flags;     // 标志
        uint32_t cylinders; // 柱面数
        uint32_t heads;     // 磁头数
        uint32_t sectors;   // 每磁道扇区数
        uint64_t total_sectors; // 总扇区数
        uint16_t bytes_per_sector; // 每扇区字节数
    } buffer;
    
    buffer.size = sizeof(buffer);
    
    __asm__ volatile (
        "int $0x13"
        : "=a"(buffer)
        : "a"(0x4800), "d"(drive), "S"(&buffer)
        : "memory"
    );
    
    return (buffer.flags & 0x01) ? 0 : -1; // 0=成功
}
```

### 2.3 键盘服务（INT 16h）

#### 键盘功能
| AH | 功能 | 返回 |
|----|------|------|
| **0x00** | 读取字符 | AL=ASCII, AH=扫描码 |
| **0x01** | 检查键盘缓冲区 | ZF=1=无字符, AX=字符 |
| **0x02** | 获取移位状态 | AL=移位状态 |

#### 键盘输入处理
```assembly
; 等待按键
wait_key:
    mov ah, 0x00
    int 0x16
    ; AL = ASCII, AH = 扫描码
    ret

; 检查是否有按键
check_key:
    mov ah, 0x01
    int 0x16
    jz no_key       ; ZF=1 表示无按键
    
    ; AX = 字符
    ret

no_key:
    xor ax, ax
    ret
```

#### 扫描码处理
```c
// 扫描码到 ASCII 转换表
static const char scan_to_ascii[128] = {
    0, 0x1B, '1', '2', '3', '4', '5', '6',
    '7', '8', '9', '0', '-', '=', '\b', '\t',
    'q', 'w', 'e', 'r', 't', 'y', 'u', 'i',
    'o', 'p', '[', ']', '\n', 0, 'a', 's',
    'd', 'f', 'g', 'h', 'j', 'k', 'l', ';',
    '\'', '`', 0, '\\', 'z', 'x', 'c', 'v',
    'b', 'n', 'm', ',', '.', '/', 0, '*',
    0, ' ', 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, '7',
    '8', '9', '-', '4', '5', '6', '+', '1',
    '2', '3', '0', '.'
};
```

---

## 第三章：BIOS 硬件检测与配置

### 3.1 内存检测

#### INT 15h, E820 内存映射
```c
// E820 内存映射条目
struct e820_entry {
    uint64_t base;      // 基地址
    uint64_t length;    // 长度
    uint32_t type;      // 类型（1=可用, 2=保留, 3=ACPI, 4=NVS）
    uint32_t reserved;
};

// 获取内存映射
int get_memory_map(struct e820_entry *map, int max_entries) {
    uint32_t cont_id = 0;
    int count = 0;
    
    do {
        __asm__ volatile (
            "int $0x15"
            : "=a"(cont_id), "=c"(map[count].type), "=d"(map[count].reserved)
            : "a"(0xE820), "b"(map[count].base), "d"(map[count].length), "c"(20), "D"(cont_id)
            : "memory"
        );
        
        if (cont_id == 0) break;
        
        // 检查是否为可用内存
        if (map[count].type == 1 && map[count].length > 0) {
            count++;
            if (count >= max_entries) break;
        }
    } while (cont_id != 0);
    
    return count;
}
```

#### 内存类型说明
| 类型 | 说明 |
|------|------|
| **1** | 可用 RAM |
| **2** | 保留（硬件使用）|
| **3** | ACPI 重声明 |
| **4** | NVS（非易失性存储）|
| **5** | 有缺陷的内存 |

### 3.2 CPU 检测

#### CPUID 指令
```c
// CPUID 结构
struct cpuid_result {
    uint32_t eax, ebx, ecx, edx;
};

// 执行 CPUID
static inline struct cpuid_result cpuid(uint32_t leaf) {
    struct cpuid_result result;
    __asm__ volatile (
        "cpuid"
        : "=a"(result.eax), "=b"(result.ebx), "=c"(result.ecx), "=d"(result.edx)
        : "a"(leaf)
        : "memory"
    );
    return result;
}

// 检测 CPU 功能
int detect_cpu_features(void) {
    struct cpuid_result result = cpuid(0x00000001);
    
    int features = 0;
    if (result.edx & (1 << 15)) features |= CPUID_FPU;
    if (result.edx & (1 << 23)) features |= CPUID_MMX;
    if (result.edx & (1 << 25)) features |= CPUID_SSE;
    if (result.ecx & (1 << 0))  features |= CPUID_SSE3;
    if (result.ecx & (1 << 9))  features |= CPUID_SSSE3;
    
    return features;
}
```

### 3.3 PCI 设备检测

#### PCI 配置空间访问
```c
// PCI 配置空间读取
uint32_t pci_read_config(uint8_t bus, uint8_t slot, uint8_t func, uint8_t offset) {
    uint32_t address = (1 << 31) | (bus << 16) | (slot << 11) | (func << 8) | (offset & 0xFC);
    
    outl(0xCF8, address);
    return inl(0xCFC + (offset & 0x3));
}

// 扫描 PCI 总线
void pci_scan(void) {
    for (int bus = 0; bus < 256; bus++) {
        for (int slot = 0; slot < 32; slot++) {
            uint16_t vendor = pci_read_config(bus, slot, 0, 0x00);
            if (vendor == 0xFFFF) continue; // 无设备
            
            uint16_t device = pci_read_config(bus, slot, 0, 0x02);
            uint32_t class_rev = pci_read_config(bus, slot, 0, 0x08);
            
            printf("PCI %02x:%02x.0 - %04x:%04x (class %06x)\n", 
                   bus, slot, vendor, device, class_rev >> 8);
        }
    }
}
```

---

## 第四章：BIOS 启动扇区开发

### 4.1 引导扇区结构

#### MBR 格式
| 偏移 | 大小 | 用途 |
|------|------|------|
| **0x0000** | 446 | 引导代码 |
| **0x01BE** | 16 | 分区表项 1 |
| **0x01CE** | 16 | 分区表项 2 |
| **0x01DE** | 16 | 分区表项 3 |
| **0x01EE** | 16 | 分区表项 4 |
| **0x01FE** | 2 | 签名（0xAA55）|

#### 最小引导扇区
```assembly
; boot.asm
bits 16
org 0x7C00

start:
    ; 设置段寄存器
    cli
    mov ax, 0x07C0
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00
    sti
    
    ; 清屏
    mov ah, 0x06
    mov al, 0x00
    mov bh, 0x07
    mov cx, 0x0000
    mov dx, 0x184F
    int 0x10
    
    ; 输出消息
    mov si, msg
    call print_string
    
    ; 无限循环
    cli
    hlt
    jmp $

print_string:
    lodsb
    or al, al
    jz .done
    mov ah, 0x0E
    mov bx, 0x0007
    int 0x10
    jmp print_string
.done:
    ret

msg: db "BIOS Bootloader!", 0

; 填充到 510 字节
times 510-($-$$) db 0

; 签名
dw 0xAA55
```

### 4.2 磁盘加载器

#### 加载后续扇区
```assembly
; 加载 32 个扇区到 0x8000
load_kernel:
    mov si, disk_error_msg
    
    ; 重置磁盘
    mov ah, 0x00
    mov dl, 0x80
    int 0x13
    jc disk_error
    
    ; 加载扇区
    mov ax, 0x0800    ; 目标段
    mov es, ax
    xor bx, bx        ; 目标偏移
    mov ah, 0x02      ; 读取功能
    mov al, 32        ; 扇区数
    mov ch, 0x00      ; 柱面
    mov cl, 0x02      ; 起始扇区
    mov dh, 0x00      ; 磁头
    mov dl, 0x80      ; 驱动器
    int 0x13
    jc disk_error
    
    ; 跳转到内核
    jmp 0x0800:0x0000

disk_error:
    call print_string
    hlt
```

### 4.3 实模式到保护模式切换

#### 启用 A20 地址线
```assembly
enable_a20:
    ; 方法 1：键盘控制器
    call a20_wait
    mov al, 0xAD
    out 0x64, al
    call a20_wait
    mov al, 0xD0
    out 0x64, al
    call a20_wait2
    in al, 0x60
    or al, 2
    call a20_wait
    mov al, 0xD1
    out 0x64, al
    call a20_wait
    mov al, al
    out 0x60, al
    call a20_wait
    mov al, 0xAE
    out 0x64, al
    call a20_wait
    
    ret

a20_wait:
    in al, 0x64
    test al, 2
    jnz a20_wait
    ret

a20_wait2:
    in al, 0x64
    test al, 1
    jz a20_wait2
    ret
```

#### 设置 GDT 并切换到保护模式
```assembly
; GDT 描述符
gdt_start:
    dq 0x0                    ; 空描述符

gdt_code:
    dw 0xFFFF                 ; 段界限 0-15
    dw 0x0                    ; 基地址 0-15
    db 0x0                    ; 基地址 16-23
    db 10011010b              ; 类型：代码段，可读
    db 11001111b              ; 粒度：4KB，32位
    db 0x0                    ; 基地址 24-31

gdt_
    dw 0xFFFF
    dw 0x0
    db 0x0
    db 10010010b              ; 数据段，可写
    db 11001111b
    db 0x0

gdt_end:

gdt_descriptor:
    dw gdt_end - gdt_start - 1
    dd gdt_start

; 切换到保护模式
switch_to_pm:
    ; 加载 GDT
    lgdt [gdt_descriptor]
    
    ; 启用保护模式
    mov eax, cr0
    or eax, 1
    mov cr0, eax
    
    ; 远跳转刷新 CS
    jmp 0x08:protected_mode_start

[bits 32]
protected_mode_start:
    ; 设置段寄存器
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    
    ; 调用 C 代码
    call KERNEL_ENTRY_POINT
```

---

## 第五章：BIOS 调试技巧

### 5.1 实模式调试

#### 使用 Bochs 调试
```bash
# bochsrc.txt
megs: 32
romimage: file=/usr/share/bochs/BIOS-bochs-latest
vgaromimage: file=/usr/share/bochs/VGABIOS-lgpl-latest
ata0-master: type=disk, path=disk.img, mode=flat
boot: disk
log: bochsout.txt
debug: 1
```

#### 调试命令
```bash
# 启动 Bochs
bochs -f bochsrc.txt

# Bochs 调试器命令
b 0x7c00        # 在引导扇区设置断点
c               # 继续执行
r               # 查看寄存器
xp /10bx 0x7c00 # 查看内存
```

### 5.2 硬件模拟调试

#### QEMU 调试
```bash
# 启用调试
qemu-system-i386 -kernel kernel.bin -hda disk.img -s -S

# GDB 连接
gdb kernel.bin
(gdb) target remote :1234
(gdb) set architecture i8086
(gdb) break *0x7c00
(gdb) continue
```

### 5.3 BIOS 仿真工具

#### 使用 coreboot 仿真
```bash
# 构建 coreboot 仿真
make menuconfig
# 选择 Hardware → Emulation → QEMU
make

# 运行仿真
./build/coreboot.rom
```

#### 使用 SeaBIOS
```bash
# 编译 SeaBIOS
git clone https://github.com/coreboot/seabios.git
make

# 在 QEMU 中使用
qemu-system-i386 -bios out/bios.bin -hda disk.img
```

---

## 第六章：BIOS 与现代启动的对比

### 6.1 BIOS vs UEFI

| 特性 | BIOS | UEFI |
|------|------|------|
| **模式** | 16 位实模式 | 32/64 位保护模式 |
| **启动速度** | 慢（POST 详细） | 快（快速启动） |
| **磁盘支持** | MBR（2TB 限制） | GPT（8ZB 限制） |
| **文件系统** | 无（直接扇区访问） | FAT32（可读文件） |
| **驱动** | 内置（16 位） | 模块化（32/64 位） |
| **安全性** | 无 | Secure Boot |

### 6.2 Legacy Boot 的优势

#### 对自制 OS 开发者的价值
- **简化启动**：无需处理复杂的 UEFI 协议
- **硬件控制**：直接访问硬件端口
- **调试友好**：实模式调试工具成熟
- **兼容性**：所有 x86 PC 支持

#### 何时转向 UEFI
- **大磁盘支持**：>2TB 磁盘
- **安全启动**：Secure Boot 需求
- **现代硬件**：UEFI-only 系统
- **性能要求**：快速启动需求

### 6.3 混合启动策略

#### BIOS 兼容模式
```c
// 检测启动方式
int detect_boot_mode(void) {
    // 检查是否在 UEFI 模式下
    if (*((uint32_t*)0x00000400) == 0x55AA) {
        return BOOT_MODE_UEFI;
    } else {
        return BOOT_MODE_BIOS;
    }
}
```

#### 双重引导扇区
```
+------------------+
|  MBR (BIOS)      |  // 446 字节引导代码
+------------------+
|  GPT 头部        |  // UEFI 兼容
+------------------+
|  分区 1          |
+------------------+
|  ...             |
+------------------+
```

---

## 结论：BIOS 的持久价值

尽管 UEFI 已成为现代标准，  
**BIOS 仍然是自制操作系统开发者的最佳起点**。

通过深入理解 BIOS：
- **掌握硬件启动流程**
- **学会实模式编程**
- **理解中断服务机制**
- **构建调试调试技能**

这些技能不仅适用于 BIOS 环境，  
更为理解 UEFI 和现代操作系统奠定坚实基础。

真正的硬件掌控，  
始于对 BIOS 的深刻理解。