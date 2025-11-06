# 存储系统-挂载篇：为自制 OS 实现文件系统挂载机制

> **“mount /dev/sda1 /mnt 时，内核如何将 ext4 文件系统接入 VFS？  
> 本文将深度解析挂载全流程，从系统调用到 fill_super，  
> 并实现根文件系统与 initramfs 支持。”**

## 引言：挂载机制的核心价值

在自制操作系统中，实现文件系统只是第一步。  
**如何让文件系统真正可用**？  
答案是 **挂载**（Mounting）—— 将文件系统实例接入 VFS 树。

挂载机制解决了三大关键问题：
1. **设备到路径的映射**：`/dev/sda1` → `/mnt`
2. **文件系统初始化**：读取超级块，构建内存结构
3. **根文件系统启动**：内核如何找到第一个 init 程序？

本文将为自制 OS 实现一个**完整、安全、可扩展**的挂载框架，  
支持多文件系统、嵌套挂载、根文件系统等核心功能。

---

## 第一章：挂载核心设计原则

### 1.1 挂载 vs 挂载点

#### 关键概念：
- **文件系统实例**：磁盘上的 ext2/ext4 文件系统
- **挂载点**：VFS 树中的目录（如 `/mnt`）
- **挂载操作**：将文件系统实例绑定到挂载点

#### 挂载后 VFS 树：
```
/
├── bin/
├── dev/
├── mnt/               ← 挂载点
│   ├── file1.txt      ← 来自 /dev/sda1
│   └── file2.txt
└── usr/
```

> ✅ **访问 `/mnt/file1.txt` 时，VFS 自动路由到 /dev/sda1 的文件系统**

### 1.2 设计目标

#### 核心目标：
1. **统一挂载接口**：`mount(dev, dir, type, flags)`
2. **支持多文件系统**：ext2、ext4、procfs 等
3. **嵌套挂载**：`/mnt` 下可再挂载 `/mnt/sub`
4. **根文件系统**：启动时挂载根文件系统

#### 约束条件：
- **单线程**：暂不考虑挂载并发
- **无动态卸载**：简化实现（后续可扩展）
- **路径解析安全**：防止挂载点循环

---

## 第二章：挂载核心数据结构

### 2.1 挂载描述符（vfs_mount）

```c
// vfs.h
struct vfs_mount {
    struct vfs_super *mnt_sb;       // 关联的超级块
    struct vfs_dentry *mnt_mountpoint; // 挂载点 dentry
    struct vfs_dentry *mnt_root;    // 文件系统根 dentry
    
    char mnt_devname[64];           // 设备名（"/dev/sda1"）
    char mnt_dirname[64];           // 挂载目录（"/mnt"）
    unsigned long mnt_flags;        // 挂载标志（MS_RDONLY, MS_NOEXEC）
    
    struct list_head mnt_list;      // 挂载链表
    struct list_head mnt_children;  // 子挂载链表头
    struct vfs_mount *mnt_parent;   // 父挂载
};
```

### 2.2 全局挂载状态

```c
// vfs.c
static struct vfs_mount *root_mount = NULL; // 根挂载
static LIST_HEAD(mount_list);              // 所有挂载链表
```

### 2.3 挂载标志

```c
#define MS_RDONLY       1    // 只读挂载
#define MS_NOSUID       2    // 忽略 suid
#define MS_NODEV        4    // 忽略设备文件
#define MS_NOEXEC       8    // 不可执行
#define MS_SYNCHRONOUS  16   // 同步写入
#define MS_REMOUNT      32   // 重新挂载
#define MS_MANDLOCK     64   // 强制锁
#define MS_NOATIME      128  // 不更新 atime
```

---

## 第三章：挂载全流程实现

### 3.1 系统调用入口

```c
// sys_vfs.c
int sys_mount(const char *dev_name, 
              const char *dir_name, 
              const char *type, 
              unsigned long flags, 
              const void *data) {
    // 1. 复制用户参数
    char k_dev[64], k_dir[64], k_type[32];
    if (copy_from_user(k_dev, dev_name, sizeof(k_dev)) ||
        copy_from_user(k_dir, dir_name, sizeof(k_dir)) ||
        copy_from_user(k_type, type, sizeof(k_type))) {
        return -1;
    }
    
    // 2. 查找文件系统类型
    struct vfs_filesystem_type *fs_type = vfs_get_fs_type(k_type);
    if (!fs_type) return -1;
    
    // 3. 查找挂载点目录
    struct vfs_dentry *mountpoint;
    if (vfs_path_lookup(k_dir, &mountpoint) < 0) {
        return -1;
    }
    
    // 4. 检查挂载点是否目录
    if (!S_ISDIR(mountpoint->d_inode->i_mode)) {
        return -1;
    }
    
    // 5. 调用文件系统 mount 函数
    struct vfs_super *sb = fs_type->mount(k_dev, (void*)data);
    if (!sb) return -1;
    
    // 6. 创建挂载描述符
    struct vfs_mount *mnt = vfs_create_mount(sb, mountpoint, k_dev, k_dir, flags);
    if (!mnt) {
        fs_type->kill_sb(sb);
        return -1;
    }
    
    return 0;
}
```

### 3.2 文件系统注册与查找

```c
// vfs.c
struct vfs_filesystem_type *vfs_get_fs_type(const char *name) {
    struct vfs_filesystem_type *type;
    for (type = vfs_fs_types; type; type = type->next) {
        if (strcmp(type->name, name) == 0) {
            return type;
        }
    }
    return NULL;
}
```

### 3.3 挂载描述符创建

```c
// vfs.c
static struct vfs_mount *vfs_create_mount(struct vfs_super *sb,
                                         struct vfs_dentry *mountpoint,
                                         const char *dev_name,
                                         const char *dir_name,
                                         unsigned long flags) {
    struct vfs_mount *mnt = kmalloc(sizeof(struct vfs_mount));
    
    // 1. 初始化挂载描述符
    mnt->mnt_sb = sb;
    mnt->mnt_mountpoint = mountpoint;
    mnt->mnt_root = sb->s_root;
    strcpy(mnt->mnt_devname, dev_name);
    strcpy(mnt->mnt_dirname, dir_name);
    mnt->mnt_flags = flags;
    
    // 2. 设置挂载树关系
    mnt->mnt_parent = NULL; // 简化：无嵌套
    INIT_LIST_HEAD(&mnt->mnt_children);
    
    // 3. 加入全局链表
    list_add_tail(&mnt->mnt_list, &mount_list);
    
    return mnt;
}
```

---

## 第四章：文件系统初始化：fill_super

### 4.1 ext2 挂载实现

```c
// fs/ext2/super.c
struct vfs_super *ext2_mount(const char *dev_name, void *data) {
    // 1. 查找块设备
    struct block_device *bdev = block_find_by_name(dev_name);
    if (!bdev) return NULL;
    
    // 2. 分配超级块内存
    struct ext2_sb_info *sbi = kmalloc(sizeof(struct ext2_sb_info));
    sbi->s_bdev = bdev;
    
    // 3. 读取并验证超级块
    struct ext2_super_block *es = kmalloc(1024);
    if (block_read(bdev, 1, es, 2) < 0) { // 从 1KB 开始读 2 块
        kfree(es);
        kfree(sbi);
        return NULL;
    }
    
    if (es->s_magic != 0xEF53) {
        kfree(es);
        kfree(sbi);
        return NULL;
    }
    
    // 4. 初始化 vfs_super
    struct vfs_super *sb = kmalloc(sizeof(struct vfs_super));
    sb->s_magic = es->s_magic;
    sb->s_fs_info = sbi;
    sb->s_op = &ext2_super_ops;
    
    // 5. 填充超级块（fill_super）
    if (ext2_fill_super(sb, es) < 0) {
        kfree(es);
        kfree(sbi);
        kfree(sb);
        return NULL;
    }
    
    kfree(es);
    return sb;
}
```

### 4.2 fill_super 核心逻辑

```c
// fs/ext2/super.c
static int ext2_fill_super(struct vfs_super *sb, struct ext2_super_block *es) {
    struct ext2_sb_info *sbi = sb->s_fs_info;
    
    // 1. 复制超级块信息
    sbi->s_blocks_count = es->s_blocks_count;
    sbi->s_inodes_count = es->s_inodes_count;
    sbi->s_blocks_per_group = es->s_blocks_per_group;
    sbi->s_inodes_per_group = es->s_inodes_per_group;
    sbi->s_log_block_size = es->s_log_block_size;
    sbi->s_first_data_block = es->s_first_data_block;
    
    // 2. 计算块大小
    sbi->s_block_size = 1024 << sbi->s_log_block_size;
    
    // 3. 读取块组描述符
    int gdt_blocks = (sbi->s_groups_count + EXT2_DESC_PER_BLOCK(sb) - 1) / 
                     EXT2_DESC_PER_BLOCK(sb);
    sbi->s_group_desc = kmalloc(gdt_blocks * sbi->s_block_size);
    block_read(sbi->s_bdev, sbi->s_first_data_block + 1, 
               sbi->s_group_desc, gdt_blocks);
    
    // 4. 读取根 inode
    struct vfs_inode *root = ext2_iget(sb, EXT2_ROOT_INO);
    if (!root) return -1;
    
    // 5. 创建根 dentry
    sb->s_root = d_obtain_root(root);
    return 0;
}
```

### 4.3 超级块操作（super_operations）

```c
// fs/ext2/super.c
static struct vfs_super_operations ext2_super_ops = {
    .alloc_inode = ext2_alloc_inode,
    .destroy_inode = ext2_destroy_inode,
    .put_super = ext2_put_super,
    .statfs = ext2_statfs,
};

static struct vfs_inode *ext2_alloc_inode(struct vfs_super *sb) {
    struct ext2_inode_info *ei = kmalloc(sizeof(struct ext2_inode_info));
    ei->vfs_inode.i_sb = sb;
    return &ei->vfs_inode;
}

static void ext2_put_super(struct vfs_super *sb) {
    struct ext2_sb_info *sbi = sb->s_fs_info;
    kfree(sbi->s_group_desc);
    kfree(sbi);
    kfree(sb);
}
```

---

## 第五章：VFS 路径解析与挂载点处理

### 5.1 路径解析中的挂载点检查

```c
// vfs.c
int vfs_path_lookup(const char *path, struct vfs_dentry **dentry) {
    struct vfs_dentry *current = (path[0] == '/') ? 
                                root_mount->mnt_root : current_task->cwd;
    
    // ... 路径组件解析循环
    
    // 在每一步检查挂载点
    struct vfs_mount *mnt = lookup_mount(current);
    if (mnt) {
        // 切换到挂载的文件系统
        current = mnt->mnt_root;
    }
    
    // ... 继续解析
}
```

### 5.2 挂载点查找

```c
// vfs.c
static struct vfs_mount *lookup_mount(struct vfs_dentry *dentry) {
    struct vfs_mount *mnt;
    list_for_each_entry(mnt, &mount_list, mnt_list) {
        if (mnt->mnt_mountpoint == dentry) {
            return mnt;
        }
    }
    return NULL;
}
```

### 5.3 挂载安全检查

#### 防止挂载循环：
```c
static bool is_mount_point_safe(struct vfs_dentry *dentry) {
    // 检查 dentry 是否已是挂载点
    if (lookup_mount(dentry)) return false;
    
    // 检查是否为根目录（禁止挂载到 /）
    if (dentry == root_mount->mnt_root) return false;
    
    return true;
}
```

---

## 第六章：根文件系统与 initramfs

### 6.1 根文件系统启动流程

#### 内核启动时序：
1. **初始化块设备**：IDE/AHCI 驱动
2. **挂载根文件系统**：`mount_root()`
3. **执行 init 进程**：`/sbin/init` 或 `/init`

#### mount_root 实现：
```c
// init/main.c
void mount_root(void) {
    // 1. 尝试挂载根设备（从启动参数获取）
    const char *root_dev = get_boot_param("root");
    if (!root_dev) root_dev = "/dev/hda1";
    
    // 2. 挂载 ext2 根文件系统
    struct vfs_super *sb = ext2_mount(root_dev, NULL);
    if (!sb) {
        panic("Failed to mount root filesystem");
    }
    
    // 3. 创建根挂载
    struct vfs_dentry *root_dentry = sb->s_root;
    root_mount = vfs_create_mount(sb, root_dentry, root_dev, "/", 0);
    
    // 4. 设置当前工作目录
    current_task->cwd = root_dentry;
}
```

### 6.2 initramfs：内存根文件系统

#### 为什么需要 initramfs？
- **根设备在模块中**：如 SATA 驱动需从磁盘加载
- **加密根文件系统**：需先解密
- **复杂初始化**：如网络启动

#### initramfs 实现：
```c
// init/initramfs.c
extern uint8_t _binary_initramfs_start;
extern uint8_t _binary_initramfs_end;

void initramfs_init(void) {
    // 1. 创建 RAM disk
    uint64_t size = &_binary_initramfs_end - &_binary_initramfs_start;
    struct block_device *ramdisk = ramdisk_create(&_binary_initramfs_start, size);
    
    // 2. 挂载 initramfs 为根
    struct vfs_super *sb = ext2_mount("ram0", NULL);
    root_mount = vfs_create_mount(sb, sb->s_root, "ram0", "/", 0);
    
    // 3. 执行 /init
    exec_init("/init");
}
```

#### 链接脚本支持：
```ld
/* linker.ld */
.initramfs : {
    _binary_initramfs_start = .;
    KEEP(*(.initramfs))
    _binary_initramfs_end = .;
}
```

### 6.3 init 进程执行

```c
// init/main.c
void exec_init(const char *path) {
    // 1. 查找 init 程序
    struct vfs_dentry *init_dentry;
    if (vfs_path_lookup(path, &init_dentry) < 0) {
        panic("No init found");
    }
    
    // 2. 加载并执行
    load_elf_and_run(init_dentry);
}
```

---

## 第七章：高级挂载特性

### 7.1 重新挂载（Remount）

#### 场景：
- **只读转读写**：`mount -o remount,rw /`
- **更新挂载选项**

#### 实现：
```c
// sys_vfs.c
if (flags & MS_REMOUNT) {
    // 查找现有挂载
    struct vfs_mount *mnt = find_mount_by_dir(dir_name);
    if (mnt) {
        mnt->mnt_flags = (mnt->mnt_flags & ~MS_RMT_MASK) | (flags & MS_RMT_MASK);
        return 0;
    }
}
```

### 7.2 绑定挂载（Bind Mount）

#### 场景：
- **目录映射**：`mount --bind /source /target`
- **容器文件系统隔离**

#### 实现：
```c
// sys_vfs.c
int sys_mount_bind(const char *source, const char *target) {
    // 1. 解析 source 和 target 路径
    struct vfs_dentry *src_dentry, *tgt_dentry;
    vfs_path_lookup(source, &src_dentry);
    vfs_path_lookup(target, &tgt_dentry);
    
    // 2. 创建绑定挂载
    struct vfs_mount *mnt = kmalloc(sizeof(struct vfs_mount));
    mnt->mnt_sb = src_dentry->d_inode->i_sb; // 复用源超级块
    mnt->mnt_root = src_dentry;              // 源 dentry 作为根
    mnt->mnt_mountpoint = tgt_dentry;        // 目标作为挂载点
    // ... 其他初始化
    
    list_add_tail(&mnt->mnt_list, &mount_list);
    return 0;
}
```

### 7.3 挂载命名空间（Mount Namespace）

#### 场景：
- **容器隔离**：每个容器有独立挂载视图
- **chroot 增强**

> 💡 **自制 OS 初期可忽略，但需预留扩展点**

---

## 结论：构建灵活的挂载框架

挂载机制是自制操作系统**存储栈的 glue layer**（粘合层）。  
通过精心设计的挂载描述符和路径解析，  
我们实现了：
- **统一挂载接口**：支持多文件系统
- **安全挂载点处理**：防止循环和非法挂载
- **根文件系统支持**：initramfs 与磁盘根
- **扩展性**：为 bind mount、命名空间预留接口

此挂载框架为后续实现 **/proc、/sys、/dev** 等虚拟文件系统奠定了坚实基础。  
真正的操作系统，始于对挂载机制的深刻理解。

---

## 附录：关键接口速查

### 挂载系统调用
```c
int sys_mount(const char *dev_name, const char *dir_name, 
              const char *type, unsigned long flags, const void *data);
```

### 文件系统注册
```c
VFS_DECLARE_FILESYSTEM(fsname) // 自动生成 mount/kill_sb
```

### 根文件系统启动
```c
void mount_root(void);        // 挂载磁盘根
void initramfs_init(void);    // 初始化 initramfs
```

### 挂载点查找
```c
struct vfs_mount *lookup_mount(struct vfs_dentry *dentry);
```

> **注**：本文所有代码均为简化实现，实际使用需添加错误处理、并发控制、安全检查等。