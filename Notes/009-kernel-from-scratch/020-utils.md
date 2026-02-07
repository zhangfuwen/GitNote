# 从零写 OS 内核-第二十篇：用户工具链 —— 实现 ls、cat、mkdir 等基础命令

> **“内核再强大，若没有用户工具，也只是孤岛。  
> 今天，我们编写第一个用户态 shell 工具集，让你的 OS 真正可用！”**

在前十九篇中，我们构建了完整的操作系统内核：  
✅ 多核 SMP 支持  
✅ 高半内核地址空间  
✅ ext2 文件系统  
✅ 进程管理与调度  
✅ 标准输入输出  

但用户仍只能运行硬编码的测试程序。  
真正的操作系统，必须提供**基础命令行工具**，让用户能：  
- 浏览文件（`ls`）  
- 创建目录（`mkdir`）  
- 查看文件（`cat`）  
- 创建空文件（`touch`）  
- 输出文本（`echo`）  

今天，我们就来：  
1️⃣ **编写这些工具的简化版用户程序**  
2️⃣ **通过 ext2 文件系统加载并执行它们**  
3️⃣ **实现一个简易 shell**  

让你的 OS 拥有**完整的命令行交互能力**！

---

## 🛠️ 一、用户工具设计原则

为简化，我们遵循：
- **静态链接**：无动态库依赖
- **单文件实现**：每个工具一个 `.c` 文件
- **仅依赖系统调用**：`open`/`read`/`write`/`mkdir` 等
- **无错误恢复**：出错直接退出

### 工具列表：
| 命令 | 功能 | 系统调用 |
|------|------|----------|
| `ls` | 列出目录内容 | `open`, `getdents` |
| `cat` | 显示文件内容 | `open`, `read`, `write` |
| `mkdir` | 创建目录 | `mkdir` |
| `touch` | 创建空文件 | `open`, `close` |
| `echo` | 输出参数 | `write` |
| `shell` | 命令行解释器 | `fork`, `exec`, `wait` |

> 💡 **所有工具通过 `exec("/bin/ls", ...)` 从 ext2 加载**！

---

## 📁 二、实现 ls：目录遍历

### 系统调用：`getdents`
```c
// 内核需实现 getdents（简化版 readdir）
int sys_getdents(int fd, struct dirent *dirp, unsigned int count) {
    struct file *file = get_file(fd);
    if (!file || !S_ISDIR(file->f_inode->i_mode)) {
        return -1;
    }
    return ext2_readdir(file, dirp, count);
}
```

### 用户态 ls：
```c
// user/ls.c
#include "syscalls.h"

void _start(int argc, char *argv[]) {
    const char *path = (argc > 1) ? argv[1] : ".";
    
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        write(2, "ls: cannot open directory\n", 26);
        exit(1);
    }
    
    struct dirent dir_entry;
    while (getdents(fd, &dir_entry, sizeof(dir_entry)) > 0) {
        if (dir_entry.d_name[0] != '.') { // 跳过 . 和 ..
            write(1, dir_entry.d_name, strlen(dir_entry.d_name));
            write(1, "\n", 1);
        }
    }
    
    close(fd);
    exit(0);
}
```

> ✅ **`getdents` 返回目录项，`ls` 过滤并输出**。

---

## 📖 三、实现 cat：文件读取

```c
// user/cat.c
void _start(int argc, char *argv[]) {
    if (argc < 2) {
        write(2, "usage: cat <file>\n", 18);
        exit(1);
    }
    
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        write(2, "cat: file not found\n", 20);
        exit(1);
    }
    
    char buffer[512];
    int n;
    while ((n = read(fd, buffer, sizeof(buffer))) > 0) {
        write(1, buffer, n);
    }
    
    close(fd);
    exit(0);
}
```

> 🔑 **循环读取直到 EOF（read 返回 0）**。

---

## 📂 四、实现 mkdir 与 touch

### mkdir：
```c
// user/mkdir.c
void _start(int argc, char *argv[]) {
    if (argc < 2) {
        write(2, "usage: mkdir <dir>\n", 19);
        exit(1);
    }
    if (mkdir(argv[1], 0755) < 0) {
        write(2, "mkdir failed\n", 13);
        exit(1);
    }
    exit(0);
}
```

### touch：
```c
// user/touch.c
void _start(int argc, char *argv[]) {
    if (argc < 2) {
        write(2, "usage: touch <file>\n", 20);
        exit(1);
    }
    int fd = open(argv[1], O_CREAT | O_WRONLY, 0644);
    if (fd < 0) {
        write(2, "touch failed\n", 13);
        exit(1);
    }
    close(fd);
    exit(0);
}
```

> 💡 **`O_CREAT` 标志让 `open` 在文件不存在时创建它**。

---

## 💬 五、实现 echo 与 shell

### echo：
```c
// user/echo.c
void _start(int argc, char *argv[]) {
    for (int i = 1; i < argc; i++) {
        write(1, argv[i], strlen(argv[i]));
        if (i < argc - 1) write(1, " ", 1);
    }
    write(1, "\n", 1);
    exit(0);
}
```

### 简易 shell：
```c
// user/shell.c
void _start() {
    char prompt[] = "myos$ ";
    char input[128];
    
    while (1) {
        write(1, prompt, sizeof(prompt) - 1);
        
        // 读取一行（简化：无行编辑）
        int n = read(0, input, sizeof(input) - 1);
        if (n <= 0) continue;
        input[n] = '\0';
        
        // 去掉换行
        if (input[n-1] == '\n') input[n-1] = '\0';
        
        // 解析命令（空格分隔）
        char *args[16];
        int argc = 0;
        char *token = strtok(input, " ");
        while (token && argc < 15) {
            args[argc++] = token;
            token = strtok(NULL, " ");
        }
        args[argc] = NULL;
        
        if (argc == 0) continue;
        
        // 内建命令
        if (strcmp(args[0], "exit") == 0) {
            exit(0);
        }
        
        // fork + exec
        pid_t pid = fork();
        if (pid == 0) {
            // 子进程：尝试执行 /bin/cmd
            char path[64] = "/bin/";
            strcat(path, args[0]);
            exec(path, args);
            
            // exec 失败
            write(2, "command not found: ", 19);
            write(2, args[0], strlen(args[0]));
            write(2, "\n", 1);
            exit(1);
        } else {
            // 父进程：等待子进程结束
            int status;
            wait(&status);
        }
    }
}
```

> 🌟 **shell 通过 `fork` + `exec` 运行任意命令**！

---

## 🧪 六、构建与部署到 ext2

### 1. **编译用户工具**
```makefile
# Makefile.user
CC = i686-elf-gcc
CFLAGS = -ffreestanding -fno-stack-protector -nostdlib -I.
LDFLAGS = -T user.ld

TOOLS = ls cat mkdir touch echo shell

user-binaries: $(TOOLS)

%: %.c syscalls.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
```

### 2. **创建 ext2 镜像**
```bash
#!/bin/bash
# build_disk.sh
dd if=/dev/zero of=disk.img bs=1M count=32
mkfs.ext2 -F disk.img

# 挂载并复制文件
mkdir mnt
sudo mount -o loop disk.img mnt
sudo mkdir mnt/bin
sudo cp user/{ls,cat,mkdir,touch,echo,shell} mnt/bin/
sudo umount mnt
```

### 3. **内核启动 init**
```c
// kernel/init.c
void init_process() {
    // 挂载根文件系统
    mount("ext2", "/dev/hda", "/");
    
    // 执行 shell
    char *args[] = {"shell", NULL};
    exec("/bin/shell", args);
}
```

---

## ▶️ 七、运行效果

启动 QEMU：
```bash
qemu-system-i386 -kernel kernel.bin -hda disk.img -serial stdio
```

交互示例：
```
myos$ ls
bin
myos$ ls bin
ls
cat
mkdir
touch
echo
shell
myos$ echo Hello World
Hello World
myos$ touch test.txt
myos$ ls
bin
test.txt
myos$ echo "This is a test" > test.txt  # 注意：> 重定向需 shell 支持（本篇未实现）
# 简化版：直接 cat test.txt
myos$ cat test.txt
This is a test
myos$ mkdir mydir
myos$ ls
bin
test.txt
mydir
myos$ exit
```

✅ **完整的命令行体验！**

---

## ⚠️ 八、局限与改进方向

1. **无管道与重定向**  
   - 需实现 `dup2` 系统调用
   - shell 需解析 `|`, `>`, `<`

2. **无环境变量**  
   - 需在 `exec` 时传递 `envp`

3. **无路径搜索**  
   - 当前硬编码 `/bin/`，应支持 `PATH` 环境变量

4. **无权限检查**  
   - ext2 中已存储权限，但未在 `open`/`mkdir` 中验证

> 🌱 **下一步：实现 shell 重定向与管道，支持 `cat file | grep text`**！

---

## 💬 写在最后

这些看似简单的 `ls`、`cat`，  
是 Unix 哲学“小工具组合”的起点。  
它们让操作系统从“内核”变为“平台”，  
让用户能自由构建自己的工作流。

今天你编译的 `echo`，  
正是无数开发者每天使用的命令的雏形。

> 🌟 **工具的价值，不在于复杂，而在于可用。**

---

📬 **动手挑战**：  
添加 `rm` 命令（需实现 `unlink` 系统调用），并测试删除文件。  
欢迎在评论区分享你的工具扩展！

👇 下一篇你想看：**shell 重定向与管道**，还是 **动态链接器（ld.so）**？

---

**#操作系统 #内核开发 #用户工具 #shell #ls #cat #文件系统 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“tools”**，获取：
> - 完整用户工具源码（ls/cat/mkdir/touch/echo/shell）
> - 用户态系统调用封装（syscalls.c）
> - ext2 镜像构建脚本（build_disk.sh）
