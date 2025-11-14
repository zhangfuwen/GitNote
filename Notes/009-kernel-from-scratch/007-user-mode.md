# 从零写 OS 内核-第七篇：用户态进程与系统调用 —— 让你的 OS 真正“可用”！

> **“内核再强大，若不能运行用户程序，也只是精致的裸机玩具。  
> 今天，我们让内核拥抱用户态，迈出成为真正操作系统的决定性一步！”**

在前六篇中，我们完成了启动、保护模式、分页、内核多任务等关键模块。  
但正如你所说：**内核态的多任务用处有限**——它无法运行第三方程序，无法隔离错误，更谈不上“操作系统”。

真正的操作系统，必须能**安全、隔离地运行用户编写的程序**，并通过**系统调用**提供服务。

今天，我们就来实现：
✅ **用户态进程创建**  
✅ **特权级切换（Ring 3 ←→ Ring 0）**  
✅ **第一个系统调用 `sys_write`**

从此，你的 OS 不再是“自说自话”，而是能**运行用户代码的服务平台**！

---

## 🔐 一、特权级（Ring）：硬件级的安全边界

x86 架构定义了 **4 个特权级（Ring 0~3）**，但现代 OS 只用两个：
- **Ring 0：内核态** → 可执行所有指令，访问所有内存
- **Ring 3：用户态** → 禁止特权指令（如 `cli`, `hlt`, `out`），内存受页表限制

### ⚠️ 用户态尝试执行特权指令？  
CPU 会触发 **#GP（General Protection Fault）**，内核可捕获并杀死进程！

> 💡 **特权级由段选择子的 RPL 和段描述符的 DPL 共同决定**。

---

## 🧱 二、准备用户态环境：GDT 与页表

### 1. 扩展 GDT：添加用户段描述符

在之前的 GDT 基础上，增加两个 DPL=3 的段：

```nasm
gdt_code_user:
    dw 0xFFFF
    dw 0x0
    db 0x0
    db 11111010b    ; Type=可执行、可读、一致代码段，DPL=3
    db 11001111b
    db 0x0

gdt_data_user:
    dw 0xFFFF
    dw 0x0
    db 0x0
    db 11110010b    ; Type=可读写数据段，DPL=3
    db 11001111b
    db 0x0
```

> 📌 **一致代码段（Conforming）** 允许低特权级调用高特权级代码（但系统调用不用它，用中断门）。

### 2. 构建用户页目录
每个用户进程需独立页目录，布局如下：
```
0x00000000 ~ 0xBFFFFFFF → 用户空间（可读写/执行）
0xC0000000 ~ 0xFFFFFFFF → 内核空间（仅 Ring 0 可访问）
```

- 用户页表只映射用户代码、数据、栈
- 内核页表项**复制到每个用户页目录**（高 1GB 共享）

> ✅ 这样既隔离用户，又避免每次切换刷新 TLB（内核映射不变）。

---

## 📦 三、加载用户程序：简易 ELF 解析

我们假设用户程序是**静态链接的 ELF**（无动态库依赖）。

### ELF 程序头关键字段：
| 字段 | 说明 |
|------|------|
| `p_type` | `PT_LOAD` 表示需加载到内存 |
| `p_vaddr` | 虚拟地址（用户空间，如 `0x8048000`）|
| `p_filesz` | 文件中段大小 |
| `p_memsz` | 内存中段大小（可能更大，用于 `.bss`）|

### 加载步骤：
1. 从磁盘/内存读取 ELF 头
2. 遍历程序头，找到 `PT_LOAD` 段
3. 为每个段分配物理页，映射到 `p_vaddr`
4. 将段内容从 ELF 文件复制到物理页
5. 初始化用户栈（如 `0xBFFFFFFF`）

> 💡 初期可将用户程序**直接链接在内核镜像中**（如 `_binary_user_bin_start`），避免磁盘驱动复杂度。

---

## 🔄 四、从内核跳转到用户态：`iret` 的魔法

要进入 Ring 3，必须使用 **`iret` 指令**（中断返回），它能同时恢复：
- `EIP`（用户入口）
- `CS:EIP`（含 RPL=3）
- `ESP`
- `SS:ESP`（含 RPL=3）
- `EFLAGS`

### 构造用户态上下文栈：
```c
void enter_user_mode(uint32_t entry, uint32_t stack_top) {
    // 模拟中断返回栈帧
    asm volatile (
        "pushl %0\n\t"          // 用户 SS (RPL=3)
        "pushl %1\n\t"          // 用户 ESP
        "pushl %2\n\t"          // EFLAGS (IF=1 允许中断)
        "pushl %3\n\t"          // 用户 CS (RPL=3)
        "pushl %4\n\t"          // 用户 EIP
        "iret"
        :
        : "i"(USER_DATA_SEG), "r"(stack_top),
          "i"(0x202), "i"(USER_CODE_SEG), "r"(entry)
        : "memory"
    );
}
```

> 🔑 **EFLAGS 必须设置 IF=1**（bit 9），否则用户态无法响应时钟中断！

---

## 📞 五、系统调用：用户态如何“呼叫”内核？

用户程序不能直接跳转到内核（特权级不允许），  
但可以通过 **软中断**（如 `int 0x80`）**主动陷入内核**。

### 步骤：
1. **用户程序**：  
   ```c
   // write(1, "Hello", 5)
   asm("int $0x80" : : "a"(4), "b"(1), "c"(msg), "d"(5));
   ```
2. **内核 IDT**：  
   设置中断门 `0x80`，DPL=3（允许用户触发），指向 `syscall_handler`
3. **内核处理**：  
   - 保存用户上下文
   - 根据 `EAX` 查系统调用表
   - 执行 `sys_write`
   - 恢复上下文，`iret` 返回用户态

### 系统调用表：
```c
typedef uint32_t (*syscall_t)(uint32_t, uint32_t, uint32_t);
syscall_t syscalls[] = {
    [0] = sys_read,
    [1] = sys_write,  // EAX=1 → sys_write
    [2] = sys_open,
    // ...
};
```

---

## 🛡️ 六、安全第一：验证用户指针！

用户传入的指针（如 `"Hello"`）可能：
- 指向内核空间（恶意攻击）
- 指向未映射区域（崩溃风险）

必须用 **`copy_from_user`** 验证：

```c
int copy_from_user(void *dest, const void *user_src, size_t len) {
    // 检查 user_src 是否在用户空间（< 0xC0000000）
    if ((uint32_t)user_src >= 0xC0000000) {
        return -1; // 非法地址
    }
    // 检查 [user_src, user_src+len) 是否全部可读
    if (!validate_user_range(user_src, len)) {
        return -1;
    }
    memcpy(dest, user_src, len);
    return 0;
}
```

> ✅ **所有系统调用参数中的指针，都必须经过验证！**

---

## 🧪 七、测试：第一个用户程序

**user.c**（用 `i686-elf-gcc` 编译为 ELF）：
```c
void _start() {
    char msg[] = "Hello from user space!\n";
    // 系统调用号 1 = sys_write, fd=1 (stdout)
    asm volatile (
        "mov $1, %%eax\n\t"
        "mov $1, %%ebx\n\t"
        "mov %0, %%ecx\n\t"
        "mov $23, %%edx\n\t"
        "int $0x80"
        :
        : "r"(msg)
        : "eax", "ebx", "ecx", "edx"
    );
    // 不退出（简单起见）
}
```

**内核中**：
```c
void kernel_main() {
    load_user_program("user.bin");
    enter_user_mode(user_entry, user_stack_top);
}
```

运行效果：
```
Hello from user space!
```

🎉 **成功！用户程序通过系统调用，安全地使用了内核服务！**

---

## ⚠️ 八、尚未完成但关键的问题

1. **用户程序如何退出？**  
   → 实现 `sys_exit`，释放资源，切换到下一个进程

2. **如何传递参数和环境变量？**  
   → 在用户栈上构造 `argc`, `argv`, `envp`

3. **动态链接、共享库？**  
   → 后续再考虑（先搞定静态）

4. **抢占式调度？**  
   → 下一篇：时钟中断 + 时间片轮转！

---

## 💬 写在最后

从内核态到用户态，  
不仅是特权级的切换，  
更是**责任的划分**——  
内核提供服务与安全边界，用户程序专注业务逻辑。

今天你实现的 `int 0x80`，  
正是 Linux、FreeBSD 等系统调用机制的起点。

> 🌟 **操作系统不是内核，而是内核与用户程序的共生体。**

---

📬 **动手挑战**：  
编写一个用户程序，调用两次 `sys_write`，分别输出不同消息。  
欢迎在评论区分享你的用户态“Hello World”！

👇 下一篇你想看：**时钟中断与抢占式调度**，还是 **进程创建（fork/exec）**？

---

**#操作系统 #内核开发 #用户态 #系统调用 #特权级 #ELF #x86 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“syscall”**，获取：
> - 完整系统调用实现代码（含 `copy_from_user`）
> - 用户态 GDT/IDT 配置模板
> - `i686-elf-gcc` 编译用户程序的 Makefile
