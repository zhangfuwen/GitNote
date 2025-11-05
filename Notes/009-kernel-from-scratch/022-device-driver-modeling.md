# 从零写 OS 内核-第二十二篇：设备模型与 devfs —— 统一管理所有硬件设备

> **“你的内核有 UART、VGA、IDE、TTY，但它们各自为政，无法统一管理。  
> 今天，我们构建设备模型，实现 devfs，让所有设备像文件一样被发现和操作！”**

在前面的篇章中，我们实现了多个硬件驱动：  
- **UART**（串口）  
- **VGA**（显卡）  
- **IDE**（硬盘）  
- **TTY**（终端）  
- **PIT**（时钟）  

但这些驱动都是**孤立的**：  
- 没有统一的设备注册机制  
- 用户无法通过 `/dev` 查看可用设备  
- 新增驱动需硬编码路径（如 `open("/dev/tty0")`）  

真正的操作系统，必须提供**统一的设备抽象**，并支持**动态设备发现**。  

今天，我们就来：  
✅ **设计设备模型（Device Model）**  
✅ **实现 devfs（设备文件系统）**  
✅ **自动挂载 `/dev` 并列出所有设备**  

让你的 OS 拥有**现代设备管理能力**！

---

## 🧩 一、为什么需要设备模型？

### 当前驱动的问题：
| 问题 | 后果 |
|------|------|
| **无统一接口** | 每个驱动实现自己的 `open`/`read` |
| **设备路径硬编码** | 用户需记住 `/dev/tty0`、`/dev/hda` |
| **无法动态发现** | 新增设备需修改内核代码 |

### 设备模型的核心思想：
> **“一切设备皆对象，一切操作皆文件”**

- **设备注册**：驱动启动时向内核注册设备
- **设备类（Class）**：按类型分组（如 tty、block、input）
- **设备文件系统（devfs）**：自动在 `/dev` 创建设备节点

> 💡 **Linux 的 sysfs + devtmpfs 正是基于此思想**！

---

## 🏗️ 二、设备模型核心数据结构

### 1. **设备结构体（device）**
```c
#define DEVICE_NAME_LEN 32

struct device {
    char name[DEVICE_NAME_LEN];     // 设备名（如 "tty0"）
    dev_t devt;                     // 设备号（主设备 << 8 | 次设备）
    struct device_driver *driver;   // 所属驱动
    void *driver_data;              // 驱动私有数据
    
    struct device *parent;          // 父设备（如 IDE 控制器）
    struct list_head children;      // 子设备列表
    struct list_head sibling;       // 兄弟设备链表
    
    struct list_head global_link;   // 全局设备链表
};
```

### 2. **设备驱动（device_driver）**
```c
struct device_driver {
    char name[32];
    int (*probe)(struct device *dev);   // 探测设备
    int (*remove)(struct device *dev);  // 移除设备
    struct file_operations *fops;       // 文件操作
};
```

### 3. **设备号管理**
```c
// 主设备号分配
#define MAJOR_TTY    4
#define MAJOR_BLOCK  3
#define MAJOR_IDE    3  // 与 block 共享，次设备区分

dev_t mkdev(unsigned int major, unsigned int minor) {
    return (major << 8) | minor;
}

unsigned int major(dev_t devt) { return devt >> 8; }
unsigned int minor(dev_t devt) { return devt & 0xFF; }
```

---

## 🔌 三、设备注册与驱动匹配

### 1. **注册设备**
```c
// 驱动初始化时调用
void device_register(struct device *dev) {
    // 1. 加入全局设备链表
    list_add_tail(&dev->global_link, &global_device_list);
    
    // 2. 通知 devfs 创建设备节点
    devfs_add_device(dev);
    
    // 3. 触发驱动 probe（如果驱动已注册）
    if (dev->driver && dev->driver->probe) {
        dev->driver->probe(dev);
    }
}
```

### 2. **TTY 设备注册示例**
```c
// drivers/tty.c
static struct device_driver tty_driver = {
    .name = "tty",
    .fops = &tty_fops,
    .probe = tty_probe,
};

void tty_init() {
    // 注册驱动
    driver_register(&tty_driver);
    
    // 创建设备
    struct device *tty0 = kmalloc(sizeof(struct device));
    strcpy(tty0->name, "tty0");
    tty0->devt = mkdev(MAJOR_TTY, 0);
    tty0->driver = &tty_driver;
    tty0->driver_data = &tty_instances[0];
    
    device_register(tty0);
}
```

### 3. **块设备注册示例（IDE）**
```c
// drivers/ide.c
void ide_init() {
    struct device *hda = kmalloc(sizeof(struct device));
    strcpy(hda->name, "hda");
    hda->devt = mkdev(MAJOR_BLOCK, 0); // 次设备 0 = hda
    hda->driver = &block_driver;
    hda->driver_data = &ide_disks[0];
    
    device_register(hda);
}
```

> ✅ **所有设备通过统一接口注册**！

---

## 📁 四、实现 devfs：设备文件系统

devfs 是一个**内存中的虚拟文件系统**，自动在 `/dev` 创建设备节点。

### 1. **devfs 超级块与 inode**
```c
struct devfs_inode_info {
    struct device *dev; // 指向注册的设备
};

struct inode *devfs_get_inode(struct super_block *sb, struct device *dev) {
    struct inode *inode = alloc_inode();
    inode->i_mode = S_IFCHR | 0666; // 字符设备
    inode->i_rdev = dev->devt;
    
    struct devfs_inode_info *info = kmalloc(sizeof(*info));
    info->dev = dev;
    inode->i_private = info;
    
    return inode;
}
```

### 2. **自动创建设备节点**
```c
void devfs_add_device(struct device *dev) {
    // 1. 获取 devfs 超级块
    struct super_block *sb = devfs_get_sb();
    
    // 2. 在根目录创建 dentry
    struct dentry *dentry = d_alloc_name(sb->s_root, dev->name);
    
    // 3. 创建 inode
    struct inode *inode = devfs_get_inode(sb, dev);
    dentry->d_inode = inode;
    
    // 4. 加入目录项
    d_add(sb->s_root, dentry);
}
```

### 3. **挂载 devfs**
```c
void devfs_init() {
    // 创建 /dev 目录
    struct inode *dev_dir = vfs_create_dir(vfs_root, "dev");
    
    // 挂载 devfs 到 /dev
    struct super_block *sb = devfs_mount(NULL, NULL);
    sb->s_root = dev_dir;
}
```

---

## 🧪 五、用户空间：通过 /dev 访问设备

### 1. **列出所有设备**
```bash
myos$ ls /dev
tty0
hda
console
null
```

### 2. **操作设备（与之前相同）**
```c
// 用户程序
int fd = open("/dev/tty0", O_RDWR); // 自动匹配 TTY 设备
write(fd, "Hello", 5);

int disk = open("/dev/hda", O_RDONLY); // 自动匹配 IDE 设备
read(disk, buffer, 512);
```

### 3. **VFS 如何找到设备？**
- `open("/dev/tty0")` → VFS 查找 dentry
- dentry 的 inode 包含 `i_rdev = mkdev(4, 0)`
- VFS 调用 `chrdev_open`，根据主设备号 4 找到 TTY 驱动
- 驱动的 `fops` 被用于后续 `read`/`write`

> 🔑 **设备号是 VFS 与驱动之间的桥梁**！

---

## 🧱 六、字符设备与块设备框架

### 1. **字符设备注册表**
```c
#define CHRDEV_MAX 256
static struct file_operations *chrdev_fops[CHRDEV_MAX];

int register_chrdev(unsigned int major, const char *name, 
                    struct file_operations *fops) {
    if (major == 0) {
        // 动态分配主设备号
        for (major = 1; major < CHRDEV_MAX; major++) {
            if (!chrdev_fops[major]) break;
        }
    }
    chrdev_fops[major] = fops;
    return major;
}
```

### 2. **块设备注册表**
```c
#define BLKDEV_MAX 256
static struct block_device_operations *blkdev_ops[BLKDEV_MAX];

int register_blkdev(unsigned int major, const char *name, 
                    struct block_device_operations *ops) {
    blkdev_ops[major] = ops;
    return 0;
}
```

### 3. **驱动注册**
```c
// TTY 驱动注册
static int __init tty_init(void) {
    register_chrdev(MAJOR_TTY, "tty", &tty_fops);
    // ... 创建设备
}

// IDE 驱动注册
static int __init ide_init(void) {
    register_blkdev(MAJOR_BLOCK, "block", &ide_bdev_ops);
    // ... 创建设备
}
```

---

## 🧪 七、测试：动态设备发现

### 内核启动日志：
```
[INIT] Registering TTY driver (major 4)
[INIT] Registering IDE driver (major 3)
[DEVFS] Created /dev/tty0
[DEVFS] Created /dev/hda
[DEVFS] Created /dev/console
[DEVFS] Created /dev/null
```

### 用户空间验证：
```bash
myos$ ls /dev
console
hda
null
tty0

myos$ echo "test" > /dev/tty0
test
```

✅ **所有设备自动出现在 `/dev`，无需硬编码路径**！

---

## ⚠️ 八、高级话题：设备树与热插拔

1. **设备树（Device Tree）**  
   - 用于描述硬件拓扑（ARM 常用）
   - 内核解析设备树，自动创建设备

2. **热插拔（Hotplug）**  
   - USB 设备插入时，动态创建 `/dev/sda`
   - 需要 netlink 事件通知用户空间

3. **udev 替代 devfs**  
   - Linux 后期用 **udev**（用户空间）替代 devfs
   - 支持更灵活的命名规则（如 `/dev/disk/by-id/...`）

> 💡 **我们的 devfs 是简化版，但核心思想一致**！

---

## 💬 写在最后

设备模型是操作系统**硬件抽象的巅峰**。  
它让驱动开发者专注硬件细节，  
让用户和应用程序以统一方式访问设备。

今天你创建的 `/dev/tty0`，  
正是 Linux 中 `/dev` 目录的简化起源。

> 🌟 **统一的抽象，是复杂系统的优雅解药。**

---

📬 **动手挑战**：  
为你的 VGA 驱动添加设备注册，并在 `/dev` 中创建 `vga` 节点。  
欢迎在评论区分享你的设备模型扩展！

👇 下一篇你想看：**udev 用户空间设备管理**，还是 **USB 驱动框架**？

---

**#操作系统 #内核开发 #设备模型 #devfs #驱动架构 #硬件抽象 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“devfs”**，获取：
> - 完整设备模型代码（device/device_driver）
> - devfs 实现（含自动节点创建）
> - 字符/块设备注册框架模板
