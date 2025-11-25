# 从零写 OS 内核-第八篇：fork 与 exec —— 让进程“繁衍”起来！

> **“一个进程只能孤独运行？那不是操作系统，那是单机游戏。  
> 今天，我们赋予进程‘生育’能力——通过 fork 复制自己，通过 exec 蜕变为新程序！”**

在上一篇中，我们成功让内核运行了**第一个用户态程序**，并通过系统调用实现安全交互。  
但这还不够：真正的操作系统必须能**动态创建、管理和回收进程**，支持 `shell` 启动任意命令，构建进程树。

而这一切的核心，就是 Unix 哲学的两大基石：  
✅ **`fork()` —— 复制当前进程**  
✅ **`exec()` —— 用新程序替换当前进程**

今天，我们就来亲手实现这两大系统调用，让你的 OS 拥有“繁衍”能力！

---

## 🧬 一、为什么需要 fork + exec？

你可能会问：**为什么不直接“加载新程序”？**

因为 Unix 的设计哲学是：  
> **“进程创建”与“程序加载”是两个独立操作。**

### 经典流程：
```c
pid_t pid = fork();        // 复制当前进程
if (pid == 0) {
    // 子进程
    exec("/bin/ls", args); // 替换为 ls 程序
} else {
    // 父进程
    wait(&status);         // 等待子进程结束
}
```

### 优势：
- **灵活性**：`fork` 后可修改环境（如重定向 stdin/stdout），再 `exec`
- **一致性**：所有进程创建都走同一套机制
- **简洁性**：`shell` 实现极简（只需 fork + exec）

> 💡 **没有 fork/exec，就没有现代 shell、管道、后台任务！**

---

## 📦 二、进程控制块（PCB）升级

我们需要为每个进程增加生命周期管理字段：

```c
typedef struct task {
    uint32_t pid;           // 进程 ID
    uint32_t parent_pid;    // 父进程 ID
    uint32_t cr3;           // 页目录物理地址
    uint32_t esp0;          // 内核栈栈顶（TSS 用，稍后讲）
    uint8_t state;          // RUNNING, ZOMBIE, SLEEPING
    int exit_code;          // 退出状态
    struct task *next;      // 进程链表
    uint8_t kernel_stack[8192];
    // ... 其他寄存器状态
} task_t;
```

> 🔑 **关键新增**：`pid`, `parent_pid`, `state`, `exit_code` —— 进程关系与状态管理的基础。

---

## 🧪 三、实现 sys_fork：复制进程

`fork()` 的核心是**复制父进程的整个执行环境**，但有两点特殊：

1. **子进程返回 0，父进程返回子 PID**
2. **共享文件描述符（暂不实现），但内存独立**

### 步骤：
1. **分配新 PCB 和 PID**
2. **复制父进程页目录**（逐页复制物理页）
3. **复制内核栈和寄存器状态**
4. **设置返回值魔法**

### 代码框架（简化版）：
```c
uint32_t sys_fork() {
    task_t *parent = current_task;
    task_t *child = alloc_task(); // 分配新 PCB

    child->pid = next_pid++;
    child->parent_pid = parent->pid;
    child->state = TASK_RUNNING;

    // 1. 复制页目录（仅用户空间部分）
    copy_page_directory(parent->cr3, child->cr3);

    // 2. 复制内核栈
    memcpy(child->kernel_stack, parent->kernel_stack, 8192);

    // 3. 修改子进程的返回值为 0
    //    （在内核栈中找到 EAX 位置，设为 0）
    uint32_t *child_eax = (uint32_t*)(child->kernel_stack + 8192 - 16);
    *child_eax = 0;

    // 4. 将子进程加入调度队列
    add_to_run_queue(child);

    return child->pid; // 父进程返回子 PID
}
```

> ⚠️ **注意**：实际需在汇编中保存完整上下文，此处为简化逻辑。

---

## 🔄 四、实现 sys_exec：程序替换

`exec()` 不创建新进程，而是**用新程序覆盖当前进程的用户空间**。

### 步骤：
1. **释放当前用户页表**（保留内核部分）
2. **加载新 ELF 到用户空间**
3. **设置新用户栈**（压入 `argc`, `argv`, 环境变量）
4. **设置新 EIP 为 ELF 入口**

### 关键：构造用户栈布局
```
高地址
  +------------------+
  |  envp[] (NULL)   |
  +------------------+
  |  "PATH=..."      |
  +------------------+
  |  argv[] (NULL)   |
  +------------------+
  |  "arg2"          |
  +------------------+
  |  "arg1"          |
  +------------------+
  |  "/bin/ls"       | ← argv[0]
  +------------------+
  |  argc = 3        | ← ESP 指向这里
低地址
```

### 代码框架：
```c
uint32_t sys_exec(const char *path, char *const argv[]) {
    task_t *task = current_task;

    // 1. 释放用户页（保留内核映射）
    free_user_pages(task->cr3);

    // 2. 加载新 ELF
    load_elf_to_user(path, task->cr3);

    // 3. 设置新用户栈
    uint32_t user_esp = setup_user_stack(argv, task->cr3);

    // 4. 修改内核栈中的 EIP/ESP，下次 iret 跳转到新程序
    task->user_esp = user_esp;
    task->user_eip = elf_entry;

    return 0; // exec 成功永不返回（除非失败）
}
```

> ✅ **exec 成功后，原进程的代码、数据、堆栈全部被替换！**

---

## ⚰️ 五、僵尸进程与 sys_wait

当进程调用 `exit()`，它不应立即释放 PCB——  
**父进程可能需要获取其退出状态**。

### 流程：
1. 子进程退出 → 进入 **ZOMBIE** 状态（保留 PCB，释放内存）
2. 父进程调用 `wait()` → 内核返回子进程 PID 和退出码
3. 内核**彻底释放子进程 PCB**

### sys_wait 实现：
```c
uint32_t sys_wait(int *status) {
    task_t *parent = current_task;
    for (task_t *child = task_list; child; child = child->next) {
        if (child->parent_pid == parent->pid && 
            child->state == TASK_ZOMBIE) {
            // 返回退出状态
            if (status) {
                copy_to_user(status, &child->exit_code, sizeof(int));
            }
            uint32_t pid = child->pid;
            free_task(child); // 彻底释放
            return pid;
        }
    }
    return -1; // 无子进程退出
}
```

> 🛡️ **`copy_to_user`**：将内核数据安全复制到用户空间指针。

---

## 🧪 六、测试：用户态 shell 雏形

**user_init.c**（PID=1 的 init 进程）：
```c
void main() {
    char *ls_args[] = {"/bin/ls", NULL};
    char *cat_args[] = {"/bin/cat", "file.txt", NULL};

    for (;;) {
        pid_t pid = fork();
        if (pid == 0) {
            // 子进程：先执行 ls
            exec("/bin/ls", ls_args);
            // 若 exec 失败，再试 cat
            exec("/bin/cat", cat_args);
            exit(1);
        } else {
            // 父进程（init）等待子进程结束
            int status;
            wait(&status);
            printf("Child exited with %d\n", status);
        }
    }
}
```

运行效果：
```
file1.txt
file2.txt
Child exited with 0
...
```

✅ **你的 OS 现在能动态运行任意用户程序了！**

---

## ⚠️ 七、优化方向（未来工作）

1. **写时复制（Copy-on-Write, CoW）**  
   → `fork` 时不复制物理页，只设为只读，写时才复制（大幅提升性能）

2. **vfork / posix_spawn**  
   → 针对 `fork`+`exec` 场景的优化

3. **进程组与会话**  
   → 支持 shell 作业控制（jobs, fg/bg）

4. **信号传递**  
   → `Ctrl+C` 杀死前台进程

---

## 💬 写在最后

`fork` 和 `exec` 看似简单，  
却承载了 Unix 系统**组合、复用、简洁**的设计哲学。

今天你实现的这两行系统调用，  
正是几十年来无数开发者构建复杂系统的基石。

> 🌟 **进程不再孤独，操作系统从此生生不息。**

---

📬 **动手挑战**：  
实现一个用户程序，调用 `fork()` 两次，创建两个子进程分别执行不同命令。  
欢迎在评论区分享你的进程树！

👇 下一篇你想看：**时钟中断与抢占式调度**，还是 **虚拟文件系统（VFS）入门**？

---

**#操作系统 #内核开发 #fork #exec #进程管理 #系统调用 #Unix #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“fork”**，获取：
> - 完整 `sys_fork` / `sys_exec` 实现代码
> - 进程 PCB 结构体详细注释
> - 用户态 init 程序模板（含 argv 构造）
