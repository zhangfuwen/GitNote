
# 从零写 OS 内核-第十篇：虚拟文件系统与 ext2 实现

> **“Initramfs 只是玩具，真正的操作系统需要持久化存储。  
> 今天，我们设计 VFS 抽象层，并实现第一个磁盘文件系统——ext2！”**

在上一篇中，我们通过内存中的 Initramfs 实现了用户程序加载，但这只是临时方案。  
真正的操作系统必须能**从磁盘读写文件**，支持持久化、大容量存储。

而 Linux 的经典文件系统 **ext2**（Second Extended Filesystem）正是绝佳的学习对象：  
✅ 结构清晰，文档齐全  
✅ 无日志（简单），适合裸机实现  
✅ 是 ext3/ext4 的基础  

今天，我们就来：  
1️⃣ **设计虚拟文件系统（VFS）抽象层**  
2️⃣ **解析 ext2 磁盘布局**  
3️⃣ **实现 `open`/`read`/`readdir` 系统调用**

让你的 OS 能真正从硬盘（或 QEMU 虚拟磁盘）加载程序！

---

## 🌉 一、为什么需要 VFS（虚拟文件系统）？

不同文件系统（ext2、FAT32、NTFS）结构千差万别，  
但用户程序只关心 `open`、`read`、`write` 等统一接口。

**VFS 的核心思想**：  
> **“用统一接口屏蔽底层差异”**

### VFS 核心对象：
| 对象 | 作用 |
|------|------|
| **Superblock** | 描述整个文件系统（块大小、inode 数等）|
| **Inode** | 描述单个文件（权限、大小、数据块指针）|
| **Dentry** | 目录项（文件名 → inode 映射）|
| **File** | 打开的文件实例（含当前读写位置）|

> 💡 **VFS 是“面向对象”思想在 C 中的经典实践**。

---

## 💾 二、ext2 磁盘布局详解

ext2 将磁盘划分为 **块组（Block Group）**，每个块组包含：

```
+-----------------+
|  超级块 (Superblock)  | ← 通常只在第 0 块组有完整副本
+-----------------+
|  块组描述符 (GDT)    |
+-----------------+
|  块位图 (Block Bitmap) |
+-----------------+
|  inode 位图 (Inode Bitmap) |
+-----------------+
|  inode 表 (Inode Table) |
+-----------------+
|  数据块 (Data Blocks)  |
+-----------------+
```

### 关键结构：

#### 1. **超级块（super_block）**
```c
struct ext2_super_block {
    uint32_t s_inodes_count;    // inode 总数
    uint32_t s_blocks_count;    // 块总数
    uint32_t s_r_blocks_count;  // 保留块数
    uint32_t s_free_blocks_count;
    uint32_t s_free_inodes_count;
    uint32_t s_first_data_block; // 第一个数据块编号（通常为 1）
    uint32_t s_log_block_size;   // 块大小 = 1024 << s_log_block_size
    uint32_t s_blocks_per_group;
    // ... 其他字段
};
```

#### 2. **Inode**
```c
struct ext2_inode {
    uint16_t i_mode;        // 文件类型与权限
    uint32_t i_size;        // 文件大小（字节）
    uint32_t i_block[15];   // 数据块指针
    // ... 其他字段
};
```

#### 3. **数据块寻址**
- **直接块**：`i_block[0~11]` → 直接指向数据块
- **间接块**：`i_block[12]` → 指向一个块，该块包含 1024 个数据块指针（4KB/4B）
- **双重间接**：`i_block[13]` → 两级间接
- **三重间接**：`i_block[14]` → 三级间接（ext2 支持最大 2TB 文件！）

> ✅ **初期只需实现直接块 + 一级间接**，即可支持 48KB + 4MB = ~4.2MB 文件。

---

## 🧱 三、VFS 核心数据结构设计

### 1. **文件系统类型注册**
```c
struct filesystem_type {
    const char *name;
    int (*mount)(struct super_block *sb, void *data);
    // ... 其他操作
};

// ext2 文件系统注册
static struct filesystem_type ext2_fs_type = {
    .name = "ext2",
    .mount = ext2_mount,
};
```

### 2. **超级块（内存中）**
```c
struct super_block {
    struct filesystem_type *s_type;
    void *s_fs_info;        // 指向 ext2_sb_info
    struct inode *s_root;   // 根目录 inode
};
```

### 3. **Inode（内存中）**
```c
struct inode {
    struct super_block *i_sb;
    uint32_t i_ino;         // inode 编号
    uint16_t i_mode;
    uint32_t i_size;
    void *i_private;        // 指向 ext2_inode
};
```

### 4. **文件描述符**
```c
struct file {
    struct inode *f_inode;
    off_t f_pos;            // 当前读写位置
    const struct file_operations *f_op;
};
```

---

## ⚙️ 四、实现 ext2 核心操作

### 1. **读取磁盘块**
```c
// 从磁盘读取一个块到缓冲区
void ext2_read_block(struct super_block *sb, uint32_t block_num, void *buf) {
    // 假设磁盘驱动已实现 block_device_read
    uint32_t block_size = EXT2_BLOCK_SIZE(sb);
    uint64_t offset = (uint64_t)block_num * block_size;
    block_device_read(offset, buf, block_size);
}
```

### 2. **从 inode 读取数据**
```c
int ext2_read_inode_data(struct inode *inode, char *buf, size_t count, off_t pos) {
    struct ext2_inode *ei = (struct ext2_inode*)inode->i_private;
    uint32_t block_size = EXT2_BLOCK_SIZE(inode->i_sb);
    uint32_t start_block = pos / block_size;
    uint32_t end_block = (pos + count + block_size - 1) / block_size;

    for (uint32_t b = start_block; b < end_block; b++) {
        uint32_t phys_block = ext2_get_data_block(inode, b);
        if (phys_block == 0) break; // 空洞

        char block_buf[4096];
        ext2_read_block(inode->i_sb, phys_block, block_buf);

        // 复制所需部分
        uint32_t offset_in_block = (b == start_block) ? (pos % block_size) : 0;
        uint32_t to_copy = min(block_size - offset_in_block, count);
        memcpy(buf, block_buf + offset_in_block, to_copy);

        buf += to_copy;
        count -= to_copy;
    }
    return original_count - count;
}
```

### 3. **解析路径（根目录 → inode）**
```c
struct inode *ext2_lookup(struct inode *dir, const char *name) {
    // 读取目录数据块
    // 目录项格式: inode_num | rec_len | name_len | type | name
    // 遍历所有目录项，匹配 name
    // 返回对应 inode
}
```

---

## 📁 五、系统调用对接 VFS

### sys_open:
```c
int sys_open(const char *pathname, int flags) {
    // 1. 从根目录开始解析路径
    struct inode *inode = vfs_path_lookup(pathname);
    if (!inode) return -1;

    // 2. 分配 file 结构
    struct file *file = alloc_file();
    file->f_inode = inode;
    file->f_pos = 0;
    file->f_op = &ext2_file_ops; // ext2 实现的 read/write

    return alloc_fd(file);
}
```

### sys_read:
```c
int sys_read(int fd, void *buf, size_t count) {
    struct file *file = get_file(fd);
    if (!file) return -1;

    // 调用具体文件系统的 read 操作
    int ret = file->f_op->read(file, buf, count);
    file->f_pos += ret;
    return ret;
}
```

---

## 🧪 六、测试：从 ext2 磁盘加载用户程序

### 步骤：
1. **创建 ext2 镜像**：
   ```bash
   dd if=/dev/zero of=disk.img bs=1M count=32
   mkfs.ext2 -F disk.img
   mkdir mnt && sudo mount -o loop disk.img mnt
   sudo cp bin/ls bin/sh mnt/
   sudo umount mnt
   ```

2. **QEMU 启动**：
   ```bash
   qemu-system-i386 -kernel kernel.bin -hda disk.img -serial stdio
   ```

3. **内核中**：
   ```c
   void init() {
       // 挂载 ext2
       mount("ext2", "/dev/hda", "/");
       // 执行用户程序
       exec("/ls", (char*[]){"ls", NULL});
   }
   ```

运行效果：
```
ls
sh
```

✅ **成功从 ext2 文件系统加载并运行程序！**

---

## ⚠️ 七、尚未实现但关键的功能

1. **写支持**：更新 inode、位图、数据块
2. **目录创建**：`mkdir`, `rmdir`
3. **权限检查**：`i_mode` 与进程 UID/GID 比较
4. **缓存层**：块缓存（Buffer Cache）提升性能

> 🌱 **下一步：实现块设备驱动（IDE/AHCI）**，让 OS 能真正访问物理硬盘！

---

## 💬 写在最后

ext2 虽然“古老”，  
但它清晰的结构、成熟的文档，  
使其成为学习文件系统的**最佳起点**。

今天你实现的 `open`/`read`，  
正是 Linux VFS 层的简化版。

> 🌟 **文件系统是操作系统的“记忆”——没有它，一切转瞬即逝。**

---

📬 **动手挑战**：  
在 ext2 镜像中添加一个新文件，并在内核启动时读取其内容。  
欢迎在评论区分享你的磁盘布局分析！

👇 下一篇你想看：**块设备驱动（IDE）**，还是 **ext2 写支持与 mkdir**？

---

**#操作系统 #内核开发 #文件系统 #ext2 #VFS #磁盘存储 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“ext2”**，获取：
> - 完整 ext2 读取实现代码
> - ext2 磁盘镜像分析工具（Python）
> - QEMU + ext2 启动脚本
