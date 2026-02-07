# 从零写 OS 内核-第六篇：初探多任务 —— 从单核循环到进程切换

> **“一个内核只能干一件事？那叫裸机程序。  
> 真正的操作系统，必须能‘同时’做多件事！”**

在前五篇中，我们完成了内核的启动、进入保护模式、开启分页，构建了虚拟内存的基础。  
但到目前为止，我们的内核仍然是**单任务**的：从 `kernel_main` 开始，一路执行到底，无法中断，无法切换。

而现代操作系统的核心能力之一就是**多任务（Multitasking）**——  
让用户感觉多个程序在“同时”运行。

今天，我们就来实现**最简化的协作式多任务**，并理解**进程上下文切换**的核心原理。  
虽不能“真并发”，但已迈出多任务的第一步！

---

## 🤔 一、什么是多任务？为什么需要它？

**多任务**是指操作系统**在多个任务（进程/线程）之间快速切换**，  
让用户产生“同时运行”的错觉（即使 CPU 只有一个核心）。

### 多任务的两种类型：
| 类型 | 说明 | 实现难度 |
|------|------|----------|
| **协作式（Cooperative）** | 任务主动让出 CPU（如调用 `yield()`） | ⭐ 简单 |
| **抢占式（Preemptive）** | 由时钟中断强制切换（如 Linux） | ⭐⭐⭐ 复杂 |

> 💡 **今天我们先实现协作式多任务**——它逻辑清晰，是理解上下文切换的最佳起点。

---

## 🧱 二、核心概念：进程与上下文

### 什么是进程？
- 一个**运行中的程序实例**
- 拥有**独立的虚拟地址空间**（我们已有分页！）
- 拥有**执行状态**：寄存器、栈、程序计数器等

### 什么是上下文（Context）？
- CPU 寄存器的**完整快照**，包括：
  - 通用寄存器（EAX, EBX, ...）
  - 指令指针（EIP）
  - 栈指针（ESP）
  - 标志寄存器（EFLAGS）
  - 段寄存器（CS, DS, ...）

> ✅ **上下文切换 = 保存当前任务状态 + 恢复下一个任务状态**

---

## 📦 三、数据结构：进程控制块（PCB）

我们需要一个结构体来描述每个进程：

```c
#define STACK_SIZE 8192

typedef struct {
    uint32_t esp;           // 栈指针（最关键！）
    uint32_t eip;           // 指令指针
    uint32_t eax, ebx, ecx, edx;
    uint32_t esi, edi, ebp;
    uint32_t eflags;
    uint32_t cr3;           // 页目录地址（支持虚拟内存隔离）
    uint8_t state;          // 0=RUNNING, 1=READY, 2=SLEEPING
    uint8_t stack[STACK_SIZE]; // 内核栈（每个进程独立！）
} task_t;

#define MAX_TASKS 8
task_t tasks[MAX_TASKS];
int current_task = 0;
int task_count = 0;
```

> 🔑 **每个进程必须有独立的内核栈**！  
> 否则切换时栈会互相覆盖，导致崩溃。

---

## ⚙️ 四、上下文切换：汇编是唯一选择

C 语言无法直接操作 `EIP`（程序计数器），**必须用汇编实现上下文保存与恢复**。

### 1. 保存当前上下文（`switch_to` 的前半部分）
```nasm
; void switch_to(task_t *next);
global switch_to
switch_to:
    ; 保存当前寄存器到 current_task 的 PCB
    push ebp
    push edi
    push esi
    push edx
    push ecx
    push ebx
    push eax

    ; 保存 ESP（当前栈顶）
    mov eax, [esp + 28]     ; 跳过 7 个 push 的 28 字节
    mov [eax], esp          ; eax 是 task_t*，esp 存入 task->esp

    ; 保存 EFLAGS
    pushf
    pop ebx
    mov [eax + 4], ebx      ; task->eflags

    ; 保存 EIP：稍后通过 ret 实现
```

### 2. 加载下一个任务上下文
```nasm
    ; eax 仍是 next_task 指针
    mov esp, [eax]          ; 切换到 next 的栈

    ; 恢复 EFLAGS
    mov ebx, [eax + 4]
    push ebx
    popf

    ; 恢复通用寄存器
    pop eax
    pop ebx
    pop ecx
    pop edx
    pop esi
    pop edi
    pop ebp

    ret                     ; 弹出 EIP，跳转到 next 任务
```

> 🌟 **`ret` 指令会从栈顶弹出 EIP**——这正是我们保存在任务栈中的“下一条指令地址”！

---

## 🔄 五、协作式调度器：实现 `yield()`

```c
void yield() {
    int next = (current_task + 1) % task_count;
    if (next != current_task) {
        current_task = next;
        switch_to(&tasks[next]);
    }
}
```

### 创建新任务：
```c
int create_task(void (*entry)()) {
    if (task_count >= MAX_TASKS) return -1;

    task_t *t = &tasks[task_count];
    t->eip = (uint32_t)entry;
    t->esp = (uint32_t)t->stack + STACK_SIZE - 8; // 栈顶

    // 初始化栈：让 switch_to 的 ret 能跳转到 entry
    *(uint32_t*)(t->esp) = t->eip;  // EIP 压栈

    task_count++;
    return task_count - 1;
}
```

> 💡 **关键技巧**：在新任务的栈顶预先压入 `entry` 地址，  
> 这样当 `switch_to` 执行 `ret` 时，会直接跳转到 `entry()`！

---

## 🧪 六、测试：让两个任务交替运行

```c
void task1() {
    int i = 0;
    while (1) {
        uart_puts("Task 1: ");
        uart_put_hex(i++);
        uart_puts("\r\n");
        yield(); // 主动让出 CPU
    }
}

void task2() {
    int j = 0;
    while (1) {
        uart_puts("Task 2: ");
        uart_put_hex(j++);
        uart_puts("\r\n");
        yield();
    }
}

void kernel_main() {
    uart_init();
    create_task(task1);
    create_task(task2);

    // 启动第一个任务
    switch_to(&tasks[0]);
}
```

运行效果：
```
Task 1: 0
Task 2: 0
Task 1: 1
Task 2: 1
...
```

✅ **成功实现多任务切换！**

---

## ⚠️ 七、尚未解决的问题（为抢占式做准备）

当前协作式多任务有明显缺陷：
1. **一个任务死循环，系统卡死**（必须主动 `yield`）
2. **无时间片概念**，无法公平调度
3. **无进程创建/销毁 API**
4. **无用户态支持**

> 🔜 下一步：**引入时钟中断（PIT/HPET）**，实现**抢占式调度**！

---

## 🌱 八、多任务的意义：不只是“看起来同时”

多任务机制为后续功能奠定基础：
- **进程隔离**：每个进程有自己的页目录（`cr3`）
- **系统调用**：用户程序通过 `int 0x80` 进入内核
- **信号与 IPC**：进程间通信
- **文件描述符表**：每个进程独立的打开文件列表

> 💡 **进程是操作系统的“基本调度单位”**，  
> 掌握它，你就真正踏入了操作系统的核心。

---

## 💬 写在最后

从单任务到多任务，  
不仅是代码结构的改变，  
更是**思维方式的跃迁**——  
从“顺序执行”到“并发抽象”。

今天你写的 `switch_to`，  
正是 Linux `context_switch`、Windows 线程调度的雏形。

> 🌟 **多任务不是魔法，而是精心设计的状态切换。**

---

📬 **动手试试**：  
在你的内核中添加两个任务，一个打印 “A”，一个打印 “B”，观察交替输出。  
欢迎在评论区晒出你的多任务日志！

👇 下一篇你想看：**时钟中断与抢占式调度**，还是 **系统调用接口设计**？

---

**#操作系统 #内核开发 #多任务 #进程管理 #上下文切换 #协作式调度 #x86 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“task”**，获取：
> - 完整多任务实现代码（含汇编上下文切换）
> - PCB 结构体详细注释
> - QEMU 调试多任务切换的 GDB 脚本
