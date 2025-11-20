
# 从零写 OS 内核-第十三篇：sysfs —— 以文件形式暴露内核信息

> **“内核运行状态藏在寄存器和内存中，用户如何查看？  
> 今天，我们实现 sysfs，让 CPU、内存、进程信息像普通文件一样可读！”**

在上一篇中，我们让标准流（stdin/stdout/stderr）在 VFS 中工作，用户程序终于能“说话”了。  
但操作系统不仅是程序的运行平台，更是**系统状态的管理者**——用户需要监控 CPU 使用率、内存占用、进程列表等。

Linux 通过 **sysfs**（`/sys`）和 **procfs**（`/proc`）将内核数据结构**以文件形式暴露给用户空间**。  
今天，我们就来实现一个**简化版 sysfs**，让你的 OS 拥有“自省”能力！

---

## 🧩 一、什么是 sysfs？为什么需要它？

**sysfs** 是一个**内存中的虚拟文件系统**，它：
- **不占用磁盘空间**
- **动态生成内容**（读文件时才计算）
- **组织为层次化目录**（反映内核对象关系）

### 典型 sysfs 路径：
| 路径 | 内容 |
|------|------|
| `/sys/cpu/0/freq` | CPU 0 频率（Hz）|
| `/sys/mem/total` | 总物理内存（KB）|
| `/sys/kernel/version` | 内核版本字符串 |
| `/sys/devices/ide/hda/model` | 磁盘型号 |

> 💡 **核心思想**：**把内核数据结构映射为文件和目录**，用户通过 `cat` 即可查看系统状态。

---

## 🏗️ 二、sysfs 架构设计

sysfs 本质是一个**特殊的文件系统类型**，注册到 VFS 中：

```c
// sysfs 文件系统类型
static struct filesystem_type sysfs_fs_type = {
    .name = "sysfs",
    .mount = sysfs_mount,
};

// 挂载点
void sysfs_init() {
    struct super_block *sb = sysfs_mount(NULL, NULL);
    // 挂载到 /sys
    vfs_mount("sysfs", "/sys", sb);
}
```

### 核心对象：sysfs_dirent
每个文件/目录对应一个 `sysfs_dirent`：

```c
enum sysfs_type {
    SYSFS_DIR,
    SYSFS_FILE,
};

struct sysfs_dirent {
    char name[32];
    enum sysfs_type type;
    struct sysfs_dirent *parent;
    struct list_head children; // 子目录/文件链表

    // 若为文件
    ssize_t (*show)(char *buf, size_t size); // 生成内容的回调
    void *private; // 私有数据（如指向 cpu_info）
};
```

> 🔑 **`show` 回调是关键**：当用户 `read` 文件时，内核调用它生成内容。

---

## ⚙️ 三、实现 sysfs 文件操作

### 1. **sysfs inode 创建**
```c
struct inode *sysfs_get_inode(struct super_block *sb, struct sysfs_dirent *sd) {
    struct inode *inode = alloc_inode();
    inode->i_mode = (sd->type == SYSFS_DIR) ? 0555 : 0444;
    inode->i_private = sd; // 指向 sysfs_dirent
    return inode;
}
```

### 2. **sysfs 文件读操作**
```c
static ssize_t sysfs_read(struct file *file, char *buf, size_t count) {
    struct inode *inode = file->f_inode;
    struct sysfs_dirent *sd = (struct sysfs_dirent*)inode->i_private;

    if (sd->type != SYSFS_FILE) return -1;

    // 调用 show 回调生成内容
    char temp_buf[256];
    ssize_t len = sd->show(temp_buf, sizeof(temp_buf));
    if (len <= 0) return len;

    // 复制到用户缓冲区（考虑偏移和长度）
    size_t to_copy = min((size_t)len - file->f_pos, count);
    if (to_copy <= 0) return 0;

    memcpy(buf, temp_buf + file->f_pos, to_copy);
    file->f_pos += to_copy;
    return to_copy;
}
```

### 3. **目录遍历（readdir）**
```c
static int sysfs_readdir(struct file *file, struct dirent *dirp) {
    struct inode *inode = file->f_inode;
    struct sysfs_dirent *sd = (struct sysfs_dirent*)inode->i_private;

    // 遍历 children 链表，返回第 file->f_pos 个条目
    struct sysfs_dirent *child = get_nth_child(sd, file->f_pos);
    if (!child) return -1;

    strcpy(dirp->d_name, child->name);
    dirp->d_ino = hash(child->name); // 简单 inode 号
    file->f_pos++;
    return 0;
}
```

---

## 📊 四、暴露系统信息：实现具体文件

### 1. **CPU 信息**
```c
// /sys/cpu/0/freq
static ssize_t cpu_freq_show(char *buf, size_t size) {
    // 读取 CPU 频率（简化：假设 1GHz）
    return snprintf(buf, size, "1000000000\n");
}

// 创建 sysfs 条目
void sysfs_create_cpu_entries() {
    struct sysfs_dirent *cpu_dir = sysfs_create_dir(NULL, "cpu");
    struct sysfs_dirent *cpu0_dir = sysfs_create_dir(cpu_dir, "0");
    sysfs_create_file(cpu0_dir, "freq", cpu_freq_show, NULL);
}
```

### 2. **内存信息**
```c
// /sys/mem/total
static ssize_t mem_total_show(char *buf, size_t size) {
    extern uint32_t total_memory_kb; // 从 Multiboot 获取
    return snprintf(buf, size, "%u\n", total_memory_kb);
}

// /sys/mem/free
static ssize_t mem_free_show(char *buf, size_t size) {
    uint32_t free = get_free_memory_kb();
    return snprintf(buf, size, "%u\n", free);
}
```

### 3. **内核版本**
```c
// /sys/kernel/version
static ssize_t kernel_version_show(char *buf, size_t size) {
    return snprintf(buf, size, "MyOS 0.1.0\n");
}
```

### 4. **进程信息（动态生成）**
```c
// /sys/processes/list
static ssize_t processes_list_show(char *buf, size_t size) {
    char *p = buf;
    for (task_t *t = task_list; t; t = t->next) {
        p += snprintf(p, buf + size - p, "PID: %d, State: %d\n", t->pid, t->state);
    }
    return p - buf;
}
```

---

## 🧪 五、测试：用户空间查看系统信息

### 内核初始化：
```c
void kernel_main() {
    // ... 其他初始化
    sysfs_init();
    sysfs_create_cpu_entries();
    sysfs_create_mem_entries();
    sysfs_create_kernel_entries();
    sysfs_create_process_entries();
}
```

### 用户程序：
```c
void _start() {
    // 读取 CPU 频率
    int fd = open("/sys/cpu/0/freq", O_RDONLY);
    char buf[32];
    read(fd, buf, sizeof(buf));
    write(1, "CPU Freq: ", 10);
    write(1, buf, strlen(buf));
    close(fd);

    // 读取进程列表
    fd = open("/sys/processes/list", O_RDONLY);
    int n = read(fd, buf, sizeof(buf));
    write(1, buf, n);
    close(fd);
}
```

运行效果：
```
CPU Freq: 1000000000
PID: 0, State: 1
PID: 1, State: 1
```

✅ **成功通过文件系统接口获取内核实时数据！**

---

## 🛠️ 六、高级特性（可选）

### 1. **可写 sysfs 文件**
- 用于**运行时配置**（如 `echo 1 > /sys/debug/enabled`）
- 实现 `store` 回调，解析用户写入的内容

```c
static ssize_t debug_store(struct file *file, const char *buf, size_t count) {
    if (count > 0 && buf[0] == '1') {
        debug_enabled = 1;
    }
    return count;
}
```

### 2. **符号链接**
- 用于创建别名（如 `/sys/cpu/current -> /sys/cpu/0`）
- 在 `sysfs_dirent` 中增加 `target` 字段

### 3. **属性组（Attribute Groups）**
- 自动为对象创建一组文件（如 CPU 对象自动有 freq/usage/temp）
- 通过宏简化注册

---

## ⚠️ 七、与 procfs 的区别

| 特性 | sysfs | procfs |
|------|-------|--------|
| **目的** | 导出**内核对象**（设备、驱动、模块）| 导出**进程信息** + 传统内核参数 |
| **结构** | 严格层次化（反映对象关系）| 扁平 + 进程子目录 |
| **现代用法** | Linux 2.6+ 主要接口 | 逐渐被 sysfs 取代（但 /proc 仍保留）|

> 💡 **我们的实现融合两者**：既有 CPU/内存（传统 procfs），又有对象层次（sysfs）。

---

## 💬 写在最后

sysfs 不仅是调试工具，  
更是**内核与用户空间对话的桥梁**。  
它让复杂的内核状态变得**直观、可脚本化、可监控**。

今天你创建的 `/sys/cpu/0/freq`，  
正是 Linux 中 `/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq` 的简化版。

> 🌟 **最好的接口，是用户早已熟悉的接口——比如文件。**

---

📬 **动手挑战**：  
添加一个新的 sysfs 文件 `/sys/kernel/uptime`，显示系统运行时间（秒）。  
欢迎在评论区分享你的 sysfs 条目设计！

👇 下一篇你想看：**procfs 实现（进程信息专用）**，还是 **动态加载内核模块（.ko 文件）**？

---

**#操作系统 #内核开发 #sysfs #虚拟文件系统 #系统监控 #VFS #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“sysfs”**，获取：
> - 完整 sysfs 实现代码（含目录/文件操作）
> - sysfs 条目注册宏（简化 API）
> - 用户态 sysfs 查看工具（cat_sysfs.c）
