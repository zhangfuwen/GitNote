# 从零写 OS 内核-第二十一篇：实现终端（Terminal）—— 让你的 shell 真正“交互”起来！

> **“你的 shell 能运行命令，但用户如何编辑命令、处理退格、响应 Ctrl+C？  
> 今天，我们实现一个完整的终端（Terminal），支持行编辑、信号和作业控制！”**

在上一篇中，我们实现了 `ls`、`cat`、`shell` 等基础命令，  
但 shell 的输入体验非常原始：  
- 无法编辑已输入的字符  
- 退格（Backspace）显示为乱码  
- 按 `Ctrl+C` 无法终止程序  
- 没有命令历史  

这一切，是因为缺少一个**终端（Terminal）**。  
终端不仅是“输入输出窗口”，更是**用户与 shell 之间的智能中介**。

今天，我们就来：  
✅ **理解终端与 TTY 的关系**  
✅ **实现内核 TTY 驱动**  
✅ **支持行编辑与基本信号**  
✅ **构建用户态终端模拟器**  

让你的 shell 拥有**现代终端体验**！

---

## 🖥️ 一、终端（Terminal）到底是什么？

### 终端的三层架构：
| 层级 | 职责 | 本篇实现 |
|------|------|----------|
| **硬件层** | 串口/VGA/USB 键盘 | 已有 UART/VGA 驱动 |
| **TTY 驱动** | 行缓冲、回显、信号处理 | ✅ 内核 TTY |
| **终端模拟器** | 光标移动、颜色、窗口 | ✅ 用户态 term |

### 关键概念：
- **TTY（Teletypewriter）**：内核中的**字符设备**（如 `/dev/tty0`）
- **终端模拟器（Terminal Emulator）**：用户态程序（如 xterm、GNOME Terminal）
- **Shell**：运行在终端上的命令解释器

> 💡 **TTY 驱动将原始硬件输入转换为结构化行数据，并处理特殊字符（如 Ctrl+C）**。

---

## 🧱 二、内核 TTY 驱动设计

TTY 驱动的核心是一个**状态机**，处理三种模式：
- **原始模式（Raw）**：直接传递字符（如 Vim）
- **规范模式（Canonical）**：行缓冲，处理退格/回车（如 Shell）
- **信号模式**：检测 Ctrl+C、Ctrl+Z 等

### TTY 数据结构：
```c
#define TTY_BUF_SIZE 1024

struct tty {
    struct file_operations fops;
    char input_buf[TTY_BUF_SIZE];
    int input_head;
    int input_tail;
    int column;             // 当前光标列（用于退格处理）
    bool echo_enabled;      // 是否回显
    bool canonical_mode;    // 是否行缓冲
    struct task *reader;    // 阻塞读取的进程
};
```

### TTY 设备注册：
```c
// 创建 /dev/tty0
void tty_init() {
    struct inode *inode = create_tty_inode(0);
    register_device("tty0", inode);
}
```

---

## ⌨️ 三、TTY 输入处理：从硬件到行缓冲

当串口收到一个字符，TTY 驱动处理逻辑如下：

### 1. **接收字符**
```c
void tty_receive_char(struct tty *tty, char c) {
    // 1. 回显（如果启用）
    if (tty->echo_enabled) {
        tty_output_char(tty, c);
    }
    
    // 2. 特殊字符处理
    if (c == '\b' || c == 127) { // Backspace
        handle_backspace(tty);
    } else if (c == '\r' || c == '\n') { // 回车
        handle_newline(tty);
    } else if (c == 3) { // Ctrl+C
        handle_sigint(tty);
    } else {
        // 普通字符：加入输入缓冲区
        tty->input_buf[tty->input_head] = c;
        tty->input_head = (tty->input_head + 1) % TTY_BUF_SIZE;
        tty->column++;
    }
}
```

### 2. **退格处理（Backspace）**
```c
void handle_backspace(struct tty *tty) {
    if (tty->column > 0) {
        // 回显：\b \s \b
        tty_output_char(tty, '\b');
        tty_output_char(tty, ' ');
        tty_output_char(tty, '\b');
        
        // 从缓冲区删除
        tty->input_head = (tty->input_head - 1 + TTY_BUF_SIZE) % TTY_BUF_SIZE;
        tty->column--;
    }
}
```

### 3. **回车处理（Newline）**
```c
void handle_newline(struct tty *tty) {
    // 添加换行符到缓冲区
    tty->input_buf[tty->input_head] = '\n';
    tty->input_head = (tty->input_head + 1) % TTY_BUF_SIZE;
    
    // 唤醒阻塞的读取进程
    if (tty->reader) {
        wake_up_process(tty->reader);
        tty->reader = NULL;
    }
}
```

### 4. **信号处理（Ctrl+C）**
```c
void handle_sigint(struct tty *tty) {
    // 回显 ^C
    tty_output_char(tty, '^');
    tty_output_char(tty, 'C');
    tty_output_char(tty, '\n');
    
    // 向前台进程组发送 SIGINT
    if (current_session && current_session->foreground_pgid) {
        kill_pg(current_session->foreground_pgid, SIGINT);
    }
    
    // 清空输入缓冲区
    tty->input_head = tty->input_tail = 0;
    tty->column = 0;
}
```

> 🔑 **TTY 驱动将原始字符流转换为“完整行”，并处理控制字符**。

---

## 📞 四、TTY 系统调用集成

TTY 作为字符设备，通过标准文件操作暴露：

### 1. **open("/dev/tty0")**
```c
int tty_open(struct inode *inode, struct file *file) {
    file->f_op = &tty_fops;
    return 0;
}
```

### 2. **read(fd, buf, count)**
```c
ssize_t tty_read(struct file *file, char *buf, size_t count) {
    struct tty *tty = get_tty_from_file(file);
    
    // 如果缓冲区无数据，阻塞等待
    while (tty->input_head == tty->input_tail) {
        tty->reader = current_task;
        sleep_on(&tty_wait_queue);
    }
    
    // 复制一行（直到 \n）
    int i = 0;
    while (i < count && tty->input_tail != tty->input_head) {
        buf[i] = tty->input_buf[tty->input_tail];
        tty->input_tail = (tty->input_tail + 1) % TTY_BUF_SIZE;
        i++;
        if (buf[i-1] == '\n') break;
    }
    
    return i;
}
```

### 3. **write(fd, buf, count)**
```c
ssize_t tty_write(struct file *file, const char *buf, size_t count) {
    for (int i = 0; i < count; i++) {
        tty_output_char(tty, buf[i]); // 直接输出到 VGA/串口
    }
    return count;
}
```

---

## 🖼️ 五、用户态终端模拟器（可选）

虽然内核 TTY 已处理行编辑，但高级终端（如支持颜色、光标移动）需用户态模拟器。

### 简易终端模拟器功能：
- **ANSI 转义序列解析**：如 `\033[2J`（清屏）
- **光标定位**：`\033[10;5H`
- **颜色支持**：`\033[31m`（红色文本）

### 用户态实现（简化）：
```c
// user/term.c
void parse_ansi(const char *buf, int len) {
    for (int i = 0; i < len; i++) {
        if (buf[i] == '\033' && i + 1 < len && buf[i+1] == '[') {
            // 解析 CSI 序列
            int j = i + 2;
            while (j < len && buf[j] != 'm' && buf[j] != 'H' && buf[j] != 'J') j++;
            if (j < len) {
                handle_ansi_sequence(buf + i + 2, j - (i + 2), buf[j]);
                i = j;
            }
        } else {
            vga_putc(buf[i]); // 普通字符
        }
    }
}
```

> 💡 **本篇重点在内核 TTY，用户态终端作为扩展**。

---

## 🧪 六、测试：交互式 shell 体验

### 用户程序：改进版 shell
```c
// user/shell.c
void _start() {
    // 打开 TTY
    int tty_fd = open("/dev/tty0", O_RDWR);
    if (tty_fd < 0) exit(1);
    
    // 设置标准输入输出为 TTY
    close(0); dup(tty_fd);
    close(1); dup(tty_fd);
    close(2); dup(tty_fd);
    close(tty_fd);
    
    char prompt[] = "myos$ ";
    while (1) {
        write(1, prompt, 7);
        char input[128];
        int n = read(0, input, sizeof(input)); // 阻塞直到回车
        if (n <= 0) continue;
        input[n] = '\0';
        if (input[n-1] == '\n') input[n-1] = '\0';
        
        // 执行命令（同上一篇）
        execute_command(input);
    }
}
```

### 运行效果：
```
myos$ ech^H^Hecho Hello
Hello
myos$ ^C
myos$ cat nonexist.txt
cat: file not found
myos$ 
```

✅ **支持退格编辑、Ctrl+C 终止、行缓冲**！

---

## ⚠️ 七、高级特性（未来方向）

1. **作业控制（Job Control）**  
   - `Ctrl+Z` 挂起进程 → `bg`/`fg` 恢复
   - 需实现进程组、会话管理

2. **命令历史**  
   - 用户态 shell 保存历史，TTY 仅提供原始输入

3. **Tab 补全**  
   - 用户态 shell 实现，TTY 透传 Tab 字符

4. **多终端支持**  
   - `/dev/tty1`, `/dev/tty2` → Alt+F1/F2 切换

> 🌱 **完整的终端体验 = 内核 TTY + 用户态 shell + 终端模拟器**

---

## 💬 写在最后

终端是用户与操作系统对话的**第一界面**。  
它看似简单，却融合了**输入处理、信号、会话管理**等核心机制。

今天你实现的 TTY 驱动，  
正是 Linux 中 `/dev/tty`、`/dev/pts/0` 的简化版。

> 🌟 **最好的交互，是让用户感觉不到技术的存在。**

---

📬 **动手挑战**：  
实现 `Ctrl+Z` 挂起功能，并添加 `jobs` 命令显示后台任务。  
欢迎在评论区分享你的终端增强功能！

👇 下一篇你想看：**作业控制（Job Control）**，还是 **伪终端（PTY）与 SSH 支持**？

---

**#操作系统 #内核开发 #终端 #TTY #行编辑 #信号处理 #shell #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“terminal”**，获取：
> - 完整 TTY 驱动代码（含退格/信号处理）
> - 用户态 shell 与 TTY 集成模板
> - ANSI 转义序列解析参考表
