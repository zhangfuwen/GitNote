# 存储系统-ext2 篇：为自制 OS 实现经典文件系统

> **“ext2 是 Linux 的经典文件系统，结构精巧、文档齐全，  
> 是自制操作系统的最佳学习起点。  
> 本文将深度解析 ext2 磁盘布局、inode 寻址、目录结构，  
> 并实现一个完整可用的 ext2 驱动。”**

## 引言：为什么选择 ext2？

在自制操作系统中，实现文件系统是**必经之路**。  
而 **ext2**（Second Extended Filesystem）是最佳选择：
- **结构清晰**：超级块、块组、位图层次分明
- **文档齐全**：官方规范详细，社区资料丰富
- **无日志**：比 ext3/ext4 简单，适合教学
- **广泛支持**：Linux、QEMU、工具链完善

本文将为自制 OS 实现一个**完整、高效、可读写**的 ext2 驱动，  
支持百万级文件组织、目录遍历、块分配等核心功能。

---

## 第一章：ext2 磁盘布局设计

### 1.1 整体布局

ext2 将磁盘划分为**块组**（Block Group），每个块组包含：

```
+-----------------+
|  超级块 (Superblock)  | ← 通常只在块组 0 有完整副本
+-----------------+
|  块组描述符 (GDT)    | ← 描述所有块组
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

#### 关键设计思想：
- **局部性**：每个块组尽量自包含，减少跨组访问
- **冗余**：超级块和 GDT 在多个块组备份
- **可扩展**：支持大磁盘（2TB）和大量文件（数百万）

### 1.2 核心数据结构

#### 超级块（ext2_super_block）
```c
// ext2.h
struct ext2_super_block {
    uint32_t s_inodes_count;    // inode 总数
    uint32_t s_blocks_count;    // 块总数
    uint32_t s_r_blocks_count;  // 保留块数
    uint32_t s_free_blocks_count;
    uint32_t s_free_inodes_count;
    uint32_t s_first_data_block; // 第一个数据块编号（通常为 1）
    uint32_t s_log_block_size;   // 块大小 = 1024 << s_log_block_size
    uint32_t s_log_frag_size;    // 片大小（通常等于块大小）
    uint32_t s_blocks_per_group;
    uint32_t s_frags_per_group;
    uint32_t s_inodes_per_group;
    uint32_t s_mtime;            // 挂载时间
    uint32_t s_wtime;            // 写入时间
    uint16_t s_mnt_count;        // 挂载次数
    uint16_t s_max_mnt_count;    // 最大挂载次数
    uint16_t s_magic;            // 魔数 = 0xEF53
    uint16_t s_state;            // 状态（EXT2_VALID_FS, EXT2_ERROR_FS）
    uint32_t s_errors;           // 错误处理行为
    uint32_t s_first_ino;        // 第一个非保留 inode（11）
    uint16_t s_inode_size;       // inode 大小（128 字节）
    // ... 其他字段
} __attribute__((packed));
```

#### 块组描述符（ext2_group_desc）
```c
struct ext2_group_desc {
    uint32_t bg_block_bitmap;       // 块位图块号
    uint32_t bg_inode_bitmap;       // inode 位图块号
    uint32_t bg_inode_table;        // inode 表起始块号
    uint16_t bg_free_blocks_count;
    uint16_t bg_free_inodes_count;
    uint16_t bg_used_dirs_count;    // 目录数
    uint16_t bg_pad;
    uint32_t bg_reserved[3];
} __attribute__((packed));
```

#### inode（ext2_inode）
```c
struct ext2_inode {
    uint16_t i_mode;        // 文件类型与权限
    uint16_t i_uid;         // 所有者 ID
    uint32_t i_size;        // 文件大小（字节）
    uint32_t i_atime;       // 访问时间
    uint32_t i_ctime;       // 创建时间
    uint32_t i_mtime;       // 修改时间
    uint32_t i_dtime;       // 删除时间
    uint16_t i_gid;         // 组 ID
    uint16_t i_links_count; // 硬链接数
    uint32_t i_blocks;      // 块数（512 字节块）
    uint32_t i_flags;       // 文件标志
    uint32_t i_osd1;        // OS 相关
    uint32_t i_block[15];   // 数据块指针
    uint32_t i_generation;  // 文件版本
    uint32_t i_file_acl;    // 访问控制列表
    uint32_t i_dir_acl;     // 目录 ACL（大文件大小）
    uint32_t i_faddr;       // 片地址
    uint32_t i_osd2[3];     // OS 相关
} __attribute__((packed));
```

---

## 第二章：inode 与数据块寻址

### 2.1 数据块寻址机制

ext2 使用 **15 个指针** 实现高效数据块寻址：

```
| i_block[0-11] | i_block[12] | i_block[13] | i_block[14] |
| 直接块        | 一级间接    | 二级间接    | 三级间接    |
```

#### 寻址能力（4KB 块大小）：
| 类型 | 指针数 | 总容量 |
|------|--------|--------|
| 直接块 | 12 | 48 KB |
| 一级间接 | 1024 | 4 MB |
| 二级间接 | 1024×1024 | 4 GB |
| 三级间接 | 1024³ | 4 TB |

> ✅ **最大文件 4TB，足够教学使用**

### 2.2 块地址解析函数

```c
// ext2.c
static uint32_t ext2_get_block_address(struct ext2_inode *inode, 
                                      uint32_t block_index) {
    struct ext2_sb_info *sb = get_sb_info();
    uint32_t block_size = EXT2_BLOCK_SIZE(sb);
    uint32_t *indirect = NULL;
    
    // 1. 直接块
    if (block_index < 12) {
        return inode->i_block[block_index];
    }
    
    // 2. 一级间接
    if (block_index < 12 + 1024) {
        if (inode->i_block[12] == 0) return 0;
        indirect = (uint32_t*)kmalloc(block_size);
        block_read(sb->s_bdev, inode->i_block[12], indirect, 1);
        uint32_t addr = indirect[block_index - 12];
        kfree(indirect);
        return addr;
    }
    
    // 3. 二级间接（简化：仅演示）
    // ... 类似实现
    
    return 0;
}
```

### 2.3 块分配与释放

#### 分配新块
```c
static uint32_t ext2_new_block(struct ext2_sb_info *sb) {
    // 1. 查找空闲块（遍历块位图）
    for (int group = 0; group < sb->s_groups_count; group++) {
        struct ext2_group_desc *gd = &sb->s_group_desc[group];
        if (gd->bg_free_blocks_count == 0) continue;
        
        // 读取块位图
        uint8_t *bitmap = kmalloc(sb->s_block_size);
        block_read(sb->s_bdev, gd->bg_block_bitmap, bitmap, 1);
        
        // 查找空闲位
        for (int i = 0; i < sb->s_blocks_per_group; i++) {
            int byte = i / 8;
            int bit = i % 8;
            if (!(bitmap[byte] & (1 << bit))) {
                // 标记为已用
                bitmap[byte] |= (1 << bit);
                block_write(sb->s_bdev, gd->bg_block_bitmap, bitmap, 1);
                
                // 更新组描述符
                gd->bg_free_blocks_count--;
                write_group_desc(sb, group);
                
                kfree(bitmap);
                return gd->bg_block_bitmap + i; // 返回绝对块号
            }
        }
        kfree(bitmap);
    }
    return 0; // 无空闲块
}
```

#### 释放块
```c
static void ext2_free_block(struct ext2_sb_info *sb, uint32_t block) {
    // 1. 计算块组和组内偏移
    uint32_t blocks_per_group = sb->s_blocks_per_group;
    int group = block / blocks_per_group;
    int offset = block % blocks_per_group;
    
    // 2. 更新块位图
    struct ext2_group_desc *gd = &sb->s_group_desc[group];
    uint8_t *bitmap = kmalloc(sb->s_block_size);
    block_read(sb->s_bdev, gd->bg_block_bitmap, bitmap, 1);
    
    int byte = offset / 8;
    int bit = offset % 8;
    bitmap[byte] &= ~(1 << bit);
    block_write(sb->s_bdev, gd->bg_block_bitmap, bitmap, 1);
    
    kfree(bitmap);
    
    // 3. 更新组描述符
    gd->bg_free_blocks_count++;
    write_group_desc(sb, group);
}
```

---

## 第三章：目录结构与文件操作

### 3.1 目录项（ext2_dir_entry）

ext2 目录是**特殊文件**，内容为目录项数组：

```c
struct ext2_dir_entry {
    uint32_t inode;         // inode 编号
    uint16_t rec_len;       // 记录长度（字节）
    uint8_t name_len;       // 名称长度
    uint8_t file_type;      // 文件类型（可选）
    char name[0];           // 可变长名称
} __attribute__((packed));
```

#### 关键设计：
- **rec_len ≥ name_len + 8**：支持记录删除（标记 inode=0）
- **rec_len 对齐**：通常 4 字节对齐
- **变长名称**：name[0] 表示动态数组

### 3.2 目录查找

```c
// ext2.c
static struct ext2_dir_entry *ext2_find_entry(struct ext2_inode *dir, 
                                             const char *name) {
    uint32_t size = dir->i_size;
    uint32_t blocks = (size + EXT2_BLOCK_SIZE(sb) - 1) / EXT2_BLOCK_SIZE(sb);
    
    for (uint32_t i = 0; i < blocks; i++) {
        uint32_t block_addr = ext2_get_block_address(dir, i);
        if (block_addr == 0) continue;
        
        char *block = kmalloc(EXT2_BLOCK_SIZE(sb));
        block_read(sb->s_bdev, block_addr, block, 1);
        
        // 遍历目录项
        char *ptr = block;
        while (ptr < block + EXT2_BLOCK_SIZE(sb)) {
            struct ext2_dir_entry *de = (void*)ptr;
            if (de->rec_len == 0) break;
            
            if (de->inode != 0 && 
                de->name_len == strlen(name) && 
                strncmp(de->name, name, de->name_len) == 0) {
                struct ext2_dir_entry *result = kmalloc(de->rec_len);
                memcpy(result, de, de->rec_len);
                kfree(block);
                return result;
            }
            
            ptr += de->rec_len;
        }
        kfree(block);
    }
    return NULL;
}
```

### 3.3 创建文件

```c
static int ext2_create(struct vfs_inode *dir, const char *name, uint16_t mode) {
    // 1. 分配新 inode
    uint32_t inode_num = ext2_new_inode(sb);
    if (inode_num == 0) return -1;
    
    // 2. 初始化 inode
    struct ext2_inode *inode = kmalloc(sizeof(struct ext2_inode));
    memset(inode, 0, sizeof(struct ext2_inode));
    inode->i_mode = mode;
    inode->i_links_count = 1;
    inode->i_uid = 0;
    inode->i_gid = 0;
    inode->i_size = 0;
    inode->i_blocks = 0;
    // ... 时间戳
    
    // 3. 写入 inode 表
    ext2_write_inode(sb, inode_num, inode);
    
    // 4. 添加目录项
    ext2_add_entry(dir, name, inode_num, mode);
    
    kfree(inode);
    return 0;
}
```

### 3.4 添加目录项

```c
static int ext2_add_entry(struct vfs_inode *dir, 
                         const char *name, 
                         uint32_t inode_num, 
                         uint16_t mode) {
    uint32_t block_size = EXT2_BLOCK_SIZE(sb);
    uint32_t dir_size = dir->i_size;
    uint32_t needed = 8 + strlen(name);
    needed = (needed + 3) & ~3; // 4 字节对齐
    
    // 1. 查找可插入的空隙
    uint32_t blocks = (dir_size + block_size - 1) / block_size;
    for (uint32_t i = 0; i < blocks; i++) {
        uint32_t block_addr = ext2_get_block_address(dir_inode, i);
        if (block_addr == 0) continue;
        
        char *block = kmalloc(block_size);
        block_read(sb->s_bdev, block_addr, block, 1);
        
        char *ptr = block;
        while (ptr < block + block_size) {
            struct ext2_dir_entry *de = (void*)ptr;
            if (de->rec_len == 0) break;
            
            // 检查空隙
            uint32_t used = 8 + de->name_len;
            uint32_t gap = de->rec_len - used;
            if (gap >= needed) {
                // 分裂记录
                struct ext2_dir_entry *new_de = (void*)(ptr + used);
                new_de->inode = inode_num;
                new_de->rec_len = gap;
                new_de->name_len = strlen(name);
                new_de->file_type = IFTODT(mode);
                strcpy(new_de->name, name);
                
                de->rec_len = used;
                
                block_write(sb->s_bdev, block_addr, block, 1);
                kfree(block);
                return 0;
            }
            
            ptr += de->rec_len;
        }
        kfree(block);
    }
    
    // 2. 无空隙，追加新块
    uint32_t new_block = ext2_new_block(sb);
    if (new_block == 0) return -1;
    
    char *new_block_data = kmalloc(block_size);
    memset(new_block_data, 0, block_size);
    
    struct ext2_dir_entry *de = (void*)new_block_data;
    de->inode = inode_num;
    de->rec_len = block_size;
    de->name_len = strlen(name);
    de->file_type = IFTODT(mode);
    strcpy(de->name, name);
    
    block_write(sb->s_bdev, new_block, new_block_data, 1);
    kfree(new_block_data);
    
    // 3. 更新目录 inode
    dir->i_size += block_size;
    dir->i_blocks += block_size / 512;
    ext2_write_inode(sb, dir->i_ino, dir);
    
    return 0;
}
```

---

## 第四章：ext2 与 VFS 集成

### 4.1 VFS 操作函数实现

#### inode_operations
```c
// ext2.c
static struct vfs_inode_operations ext2_dir_inode_ops = {
    .lookup = ext2_lookup,
    .create = ext2_create,
    .mkdir = ext2_mkdir,
    .rmdir = ext2_rmdir,
    .unlink = ext2_unlink,
};

static struct vfs_inode_operations ext2_file_inode_ops = {
    .setattr = ext2_setattr,
};
```

#### file_operations
```c
static struct vfs_file_operations ext2_file_ops = {
    .read = ext2_file_read,
    .write = ext2_file_write,
    .open = ext2_file_open,
    .release = ext2_file_release,
};

static struct vfs_file_operations ext2_dir_ops = {
    .read = ext2_dir_read,
    .open = ext2_dir_open,
    .release = ext2_dir_release,
};
```

### 4.2 文件读取实现

```c
static ssize_t ext2_file_read(struct vfs_file *file, char *buf, size_t count) {
    struct ext2_inode *inode = file->f_inode->i_private;
    uint32_t pos = file->f_pos;
    uint32_t size = inode->i_size;
    
    if (pos >= size) return 0;
    if (pos + count > size) count = size - pos;
    
    uint32_t block_size = EXT2_BLOCK_SIZE(sb);
    uint32_t start_block = pos / block_size;
    uint32_t end_block = (pos + count + block_size - 1) / block_size;
    
    char *buffer = buf;
    for (uint32_t b = start_block; b < end_block; b++) {
        uint32_t block_addr = ext2_get_block_address(inode, b);
        if (block_addr == 0) {
            memset(buffer, 0, block_size);
        } else {
            char *block = kmalloc(block_size);
            block_read(sb->s_bdev, block_addr, block, 1);
            memcpy(buffer, block, block_size);
            kfree(block);
        }
        
        buffer += block_size;
    }
    
    // 复制所需部分
    uint32_t offset_in_block = pos % block_size;
    memcpy(buf, buf + offset_in_block, count);
    
    return count;
}
```

### 4.3 目录读取实现

```c
static ssize_t ext2_dir_read(struct vfs_file *file, char *buf, size_t count) {
    // 简化：返回 dirent 结构
    struct ext2_inode *dir = file->f_inode->i_private;
    // ... 遍历目录项，填充 struct dirent
    return 0;
}
```

---

## 第五章：ext2 优化与 ext3 对比

### 5.1 块分配策略优化

#### 当前限制：
- **顺序扫描位图**：O(n) 时间
- **无预分配**：频繁分配小文件导致碎片

#### 优化方向：
1. **空闲块缓存**：缓存每个块组的空闲块范围
2. **目录局部性**：目录 inode 和数据块尽量同组
3. **预分配**：大文件分配连续块

### 5.2 ext3 日志机制对比

#### ext2 的致命缺陷：
- **元数据不一致**：写入过程中断电 → 文件系统损坏
- **fsck 慢**：启动时需全盘检查

#### ext3 解决方案：
- **Journaling**（日志）：先写日志，再写数据
- **三种模式**：
  - **journal**：元数据+数据都写日志（最安全）
  - **ordered**：元数据写日志，数据先写（默认）
  - **writeback**：仅元数据写日志（最快）

#### 日志结构：
```
+-----------------+
|  Journal Superblock |
+-----------------+
|  Descriptor Block   | ← 描述要写入的块
+-----------------+
|  Data Blocks        | ← 实际数据
+-----------------+
|  Commit Block       | ← 提交标记
+-----------------+
```

> 💡 **自制 OS 初期无需实现日志，但需注意 ext2 的脆弱性**

---

## 结论：构建可靠的存储基石

ext2 是自制操作系统的**理想文件系统起点**。  
通过实现其核心组件，  
我们掌握了：
- **磁盘布局**：超级块、块组、位图
- **数据寻址**：直接/间接块
- **目录管理**：变长目录项
- **块分配**：位图管理

此 ext2 驱动为后续实现 **ext3 日志、ext4 extents** 奠定了坚实基础。  
真正的文件系统，始于对磁盘布局的深刻理解。

---

## 附录：关键常量与工具

### ext2 常量
```c
#define EXT2_SUPER_MAGIC 0xEF53
#define EXT2_ROOT_INO 2
#define EXT2_GOOD_OLD_FIRST_INO 11

// 文件类型
#define S_IFDIR  0x4000
#define S_IFREG  0x8000
#define S_IFLNK  0xA000

// 目录项类型
#define EXT2_FT_DIR 2
#define EXT2_FT_REG_FILE 1
```

### 实用工具
- **mkfs.ext2**：创建 ext2 镜像
  ```bash
  dd if=/dev/zero of=disk.img bs=1M count=32
  mkfs.ext2 -F disk.img
  ```
- **debugfs**：调试 ext2 文件系统
  ```bash
  debugfs disk.img
  debugfs> ls
  debugfs> stat <2>  # 查看根 inode
  ```

> **注**：本文所有代码均为简化实现，实际使用需添加错误处理、边界检查、缓存等。