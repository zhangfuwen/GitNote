
# 存储系统-块设备篇：为自制 OS 设计块设备抽象层

> **“文件系统需要读写磁盘，但 IDE、SATA、NVMe 接口各不相同，  
> 如何设计统一的块设备抽象层？  
> 本文将实现通用块设备接口、MBR/GPT 分区解析、请求队列，  
> 为文件系统提供统一存储后端。”**

## 引言：块设备抽象的必要性

在自制操作系统中，初期可能只支持 **IDE 硬盘**。  
但随着系统演进，你将需要支持：
- **SATA 硬盘**（通过 AHCI 驱动）
- **NVMe SSD**（PCIe 接口）
- **RAM Disk**（内存模拟磁盘）
- **虚拟磁盘**（QEMU 模拟）

如果文件系统直接调用 `ide_read`、`ahci_read`，  
系统将** tightly coupled **（紧耦合）到特定硬件，  
扩展新设备需修改所有文件系统。

**块设备抽象层**正是为解决此问题而生！  
它通过**统一接口**，让上层（文件系统）无需关心底层硬件细节。

本文将为自制 OS 设计一个**简洁高效、支持分区**的块设备框架。

---

## 第一章：块设备核心设计原则

### 1.1 块设备 vs 字符设备

#### 关键区别：
| 特性 | 块设备 | 字符设备 |
|------|--------|----------|
| **访问单位** | 固定大小块（512B/4KB） | 字节流 |
| **随机访问** | 支持（按块偏移） | 通常不支持 |
| **缓存** | 有（块缓存） | 通常无 |
| **典型设备** | 硬盘、SSD、CD-ROM | 串口、键盘、VGA |

> ✅ **文件系统只与块设备交互**！

### 1.2 设计目标

#### 核心目标：
1. **统一接口**：文件系统调用 `block_read(bdev, sector, buf, count)`
2. **支持分区**：自动解析 MBR/GPT，创建分区设备
3. **请求合并**：合并相邻读写请求，提升性能
4. **设备无关**：IDE/AHCI/NVMe 实现统一接口

#### 约束条件：
- **扇区大小**：512 字节（传统）或 4KB（高级格式）
- **最大请求**：64 个扇区（简化）
- **单队列**：暂不实现多队列（blk-mq）

### 1.3 架构概览

```
+------------------+
|   文件系统       |  // ext2_read_block
+------------------+
|   块缓存层       |  // 可选，本文暂不实现
+------------------+
|   通用块层       |  // block_read, block_write
+------------------+
|   分区层         |  // 解析 MBR/GPT，映射分区偏移
+------------------+
|   驱动层         |  // ide, ahci, nvme, ramdisk
+------------------+
```

---

## 第二章：块设备核心数据结构

### 2.1 块设备描述符（block_device）

```c
// block.h
#define BLOCK_SECTOR_SIZE 512
#define MAX_PARTITIONS 16

struct block_operations {
    int (*submit_request)(struct block_device *bdev, 
                          uint64_t sector, 
                          void *buffer, 
                          uint32_t count, 
                          bool write);
    int (*get_capacity)(struct block_device *bdev); // 返回扇区数
};

struct block_device {
    char name[32];                      // 设备名（"hda", "sda"）
    uint32_t flags;                     // BDF_REMOVABLE, BDF_RO
    
    struct block_operations *ops;       // 块操作函数
    void *private_data;                 // 驱动私有数据
    
    // 分区信息
    struct block_device *partitions[MAX_PARTITIONS];
    uint8_t nr_partitions;
    
    // 通用信息
    uint64_t sector_size;               // 扇区大小（512/4096）
    uint64_t capacity;                  // 总扇区数
    uint8_t major, minor;               // 设备号
};
```

### 2.2 请求结构（block_request）

```c
// 简化版：同步请求（无队列）
struct block_request {
    uint64_t sector;        // 起始扇区
    void *buffer;           // 缓冲区
    uint32_t count;         // 扇区数
    bool write;             // true=写, false=读
    int result;             // 结果（0=成功）
};
```

> 💡 **为简化，本文使用同步 I/O，不实现请求队列**。  
> （高级版可扩展为异步 + 电梯算法）

---

## 第三章：通用块层实现

### 3.1 块设备注册

#### 设备号分配
```c
// block.c
#define MAJOR_IDE  3
#define MAJOR_SCSI 8
#define MAJOR_LOOP 7

static uint8_t next_minor = 0;

struct block_device *block_register_device(const char *name, 
                                          uint8_t major,
                                          struct block_operations *ops,
                                          void *private_data) {
    struct block_device *bdev = kmalloc(sizeof(struct block_device));
    strcpy(bdev->name, name);
    bdev->major = major;
    bdev->minor = next_minor++;
    bdev->ops = ops;
    bdev->private_data = private_data;
    
    // 获取容量
    bdev->capacity = ops->get_capacity(bdev);
    bdev->sector_size = BLOCK_SECTOR_SIZE;
    
    // 加入全局设备链表
    list_add_tail(&bdev->list, &block_device_list);
    
    // 自动解析分区
    partition_scan(bdev);
    
    return bdev;
}
```

### 3.2 通用读写接口

```c
// block.c
int block_read(struct block_device *bdev, 
               uint64_t sector, 
               void *buffer, 
               uint32_t count) {
    // 1. 边界检查
    if (sector + count > bdev->capacity) {
        return -1;
    }
    
    // 2. 调用驱动提交请求
    return bdev->ops->submit_request(bdev, sector, buffer, count, false);
}

int block_write(struct block_device *bdev, 
                uint64_t sector, 
                const void *buffer, 
                uint32_t count) {
    if (sector + count > bdev->capacity) {
        return -1;
    }
    return bdev->ops->submit_request(bdev, sector, (void*)buffer, count, true);
}
```

---

## 第四章：分区表解析

### 4.1 MBR 分区表解析

#### MBR 结构（512 字节）
```c
// partition.h
struct mbr_partition {
    uint8_t status;         // 0x80=active, 0x00=inactive
    uint8_t chs_first[3];   // CHS 地址（已过时）
    uint8_t type;           // 分区类型（0x83=Linux, 0x05=Extended）
    uint8_t chs_last[3];
    uint32_t lba_start;     // 起始 LBA（扇区）
    uint32_t sectors;       // 扇区数
} __attribute__((packed));

struct mbr_boot_sector {
    uint8_t boot_code[446];
    struct mbr_partition partitions[4];
    uint16_t signature;     // 0xAA55
} __attribute__((packed));
```

#### MBR 解析函数
```c
// partition.c
static void parse_mbr(struct block_device *bdev) {
    struct mbr_boot_sector *mbr = kmalloc(512);
    
    // 1. 读取 MBR
    if (block_read(bdev, 0, mbr, 1) < 0) {
        kfree(mbr);
        return;
    }
    
    // 2. 验证签名
    if (mbr->signature != 0xAA55) {
        kfree(mbr);
        return;
    }
    
    // 3. 解析主分区
    for (int i = 0; i < 4; i++) {
        if (mbr->partitions[i].type == 0) continue;
        
        // 创建分区设备
        char part_name[32];
        snprintf(part_name, sizeof(part_name), "%sp%d", bdev->name, i+1);
        
        struct partition_info *part_info = kmalloc(sizeof(struct partition_info));
        part_info->parent = bdev;
        part_info->start_sector = mbr->partitions[i].lba_start;
        part_info->sector_count = mbr->partitions[i].sectors;
        
        struct block_device *part = block_register_device(
            part_name, bdev->major, &partition_ops, part_info);
        bdev->partitions[bdev->nr_partitions++] = part;
    }
    
    kfree(mbr);
}
```

#### 分区操作函数
```c
// partition.c
static int partition_submit_request(struct block_device *bdev, 
                                   uint64_t sector, 
                                   void *buffer, 
                                   uint32_t count, 
                                   bool write) {
    struct partition_info *part = bdev->private_data;
    // 映射到父设备偏移
    return block_submit_request(part->parent, 
                                part->start_sector + sector, 
                                buffer, count, write);
}
```

### 4.2 GPT 分区表解析（简化版）

#### GPT 结构
- **LBA 0**：保护 MBR
- **LBA 1**：GPT 头部
- **LBA 2-33**：分区表（128 项 × 128 字节）

#### GPT 解析关键字段
```c
struct gpt_header {
    char signature[8];      // "EFI PART"
    uint32_t revision;
    uint32_t header_size;
    uint32_t header_crc32;
    uint32_t reserved;
    uint64_t current_lba;
    uint64_t backup_lba;
    uint64_t first_usable_lba;
    uint64_t last_usable_lba;
    uint8_t disk_guid[16];
    uint64_t partition_entries_lba;
    uint32_t num_partition_entries;
    uint32_t sizeof_partition_entry;
    uint32_t partition_entry_array_crc32;
} __attribute__((packed));

struct gpt_partition_entry {
    uint8_t partition_type_guid[16];
    uint8_t unique_partition_guid[16];
    uint64_t starting_lba;
    uint64_t ending_lba;
    uint64_t attributes;
    uint16_t name[36];      // UTF-16
} __attribute__((packed));
```

#### GPT 解析流程
```c
static void parse_gpt(struct block_device *bdev) {
    struct gpt_header *gpt = kmalloc(512);
    
    // 1. 读取 GPT 头部（LBA 1）
    if (block_read(bdev, 1, gpt, 1) < 0) {
        kfree(gpt);
        return;
    }
    
    // 2. 验证签名
    if (memcmp(gpt->signature, "EFI PART", 8) != 0) {
        kfree(gpt);
        return;
    }
    
    // 3. 读取分区表（简化：只读前 4 个）
    struct gpt_partition_entry *entries = kmalloc(512);
    block_read(bdev, gpt->partition_entries_lba, entries, 1);
    
    for (int i = 0; i < 4; i++) {
        if (entries[i].starting_lba == 0) break;
        
        char part_name[32];
        snprintf(part_name, sizeof(part_name), "%sp%d", bdev->name, i+1);
        
        struct partition_info *part_info = kmalloc(sizeof(struct partition_info));
        part_info->parent = bdev;
        part_info->start_sector = entries[i].starting_lba;
        part_info->sector_count = entries[i].ending_lba - entries[i].starting_lba + 1;
        
        struct block_device *part = block_register_device(
            part_name, bdev->major, &partition_ops, part_info);
        bdev->partitions[bdev->nr_partitions++] = part;
    }
    
    kfree(entries);
    kfree(gpt);
}
```

### 4.3 分区扫描入口

```c
// partition.c
void partition_scan(struct block_device *bdev) {
    // 1. 尝试 GPT（检查 LBA 1 签名）
    struct gpt_header test_gpt;
    if (block_read(bdev, 1, &test_gpt, 1) == 0 &&
        memcmp(test_gpt.signature, "EFI PART", 8) == 0) {
        parse_gpt(bdev);
        return;
    }
    
    // 2. 回退到 MBR
    parse_mbr(bdev);
}
```

---

## 第五章：驱动层实现示例

### 5.1 IDE 驱动适配

```c
// drivers/ide.c
static int ide_submit_request(struct block_device *bdev, 
                              uint64_t sector, 
                              void *buffer, 
                              uint32_t count, 
                              bool write) {
    struct ide_device *ide = bdev->private_data;
    
    // 调用 IDE 读写函数（前文已实现）
    if (write) {
        return ide_write_sectors(ide, sector, count, buffer);
    } else {
        return ide_read_sectors(ide, sector, count, buffer);
    }
}

static int ide_get_capacity(struct block_device *bdev) {
    struct ide_device *ide = bdev->private_data;
    return ide->capacity; // 扇区数
}

// IDE 初始化时注册
void ide_init_device(struct ide_device *ide) {
    char name[32];
    snprintf(name, sizeof(name), "hd%c", 'a' + ide->unit);
    
    static struct block_operations ide_ops = {
        .submit_request = ide_submit_request,
        .get_capacity = ide_get_capacity,
    };
    
    ide->bdev = block_register_device(name, MAJOR_IDE, &ide_ops, ide);
}
```

### 5.2 RAM Disk 驱动

```c
// drivers/ramdisk.c
struct ramdisk_device {
    void *data;
    uint64_t size; // 字节
};

static int ramdisk_submit_request(struct block_device *bdev, 
                                  uint64_t sector, 
                                  void *buffer, 
                                  uint32_t count, 
                                  bool write) {
    struct ramdisk_device *rd = bdev->private_data;
    uint64_t offset = sector * BLOCK_SECTOR_SIZE;
    uint64_t length = count * BLOCK_SECTOR_SIZE;
    
    if (offset + length > rd->size) {
        return -1;
    }
    
    if (write) {
        memcpy(rd->data + offset, buffer, length);
    } else {
        memcpy(buffer, rd->data + offset, length);
    }
    return 0;
}

static int ramdisk_get_capacity(struct block_device *bdev) {
    struct ramdisk_device *rd = bdev->private_data;
    return rd->size / BLOCK_SECTOR_SIZE;
}

// 创建 RAM Disk
struct block_device *ramdisk_create(void *data, uint64_t size) {
    struct ramdisk_device *rd = kmalloc(sizeof(struct ramdisk_device));
    rd->data = data;
    rd->size = size;
    
    static struct block_operations ramdisk_ops = {
        .submit_request = ramdisk_submit_request,
        .get_capacity = ramdisk_get_capacity,
    };
    
    return block_register_device("ram0", MAJOR_LOOP, &ramdisk_ops, rd);
}
```

---

## 第六章：VFS 与块设备集成

### 6.1 文件系统使用块设备

#### ext2 初始化示例
```c
// fs/ext2/super.c
struct vfs_super *ext2_mount(const char *dev_name, void *data) {
    // 1. 查找块设备
    struct block_device *bdev = block_find_by_name(dev_name);
    if (!bdev) return NULL;
    
    // 2. 读取超级块
    struct ext2_super_block *es = kmalloc(1024);
    if (block_read(bdev, 2, es, 2) < 0) { // ext2 超级块在 1KB-2KB
        kfree(es);
        return NULL;
    }
    
    // 3. 验证魔数
    if (es->s_magic != 0xEF53) {
        kfree(es);
        return NULL;
    }
    
    // 4. 创建 vfs_super
    struct vfs_super *sb = kmalloc(sizeof(struct vfs_super));
    sb->s_magic = 0xEF53;
    sb->s_fs_info = es;
    sb->s_bdev = bdev; // 保存块设备引用
    
    // ... 其他初始化
    return sb;
}
```

### 6.2 块设备全局查找

```c
// block.c
struct block_device *block_find_by_name(const char *name) {
    struct block_device *bdev;
    list_for_each_entry(bdev, &block_device_list, list) {
        if (strcmp(bdev->name, name) == 0) {
            return bdev;
        }
        
        // 检查分区
        for (int i = 0; i < bdev->nr_partitions; i++) {
            if (strcmp(bdev->partitions[i]->name, name) == 0) {
                return bdev->partitions[i];
            }
        }
    }
    return NULL;
}
```

---

## 第七章：高级特性展望

### 7.1 请求队列与电梯算法

#### 当前限制：
- **同步 I/O**：每次请求阻塞
- **无合并**：相邻请求不合并

#### 优化方向：
1. **请求队列**：收集多个请求
2. **电梯算法**：按扇区顺序处理（减少磁头移动）
3. **异步 I/O**：回调通知完成

```c
// 未来扩展
struct request_queue {
    struct list_head queue;
    spinlock_t lock;
    void (*make_request)(struct request_queue *q, struct block_request *req);
};

int block_make_request(struct block_device *bdev, struct block_request *req) {
    struct request_queue *q = bdev->queue;
    spin_lock(&q->lock);
    list_add_tail(&req->queuelist, &q->queue);
    spin_unlock(&q->lock);
    
    q->make_request(q, req); // 提交到驱动
    return 0;
}
```

### 7.2 多队列 blk-mq（高级）

#### 现代 SSD 优化：
- **每个 CPU 核心一个队列**
- **硬件队列映射**
- **无锁提交**

> 💡 **自制 OS 初期无需实现，但需预留扩展点**

### 7.3 块缓存层

#### 性能优化：
- **缓存常用块**：减少磁盘 I/O
- **写回策略**：延迟写入
- **LRU 回收**：内存压力时释放

```c
// 未来扩展
struct buffer_head {
    uint64_t block_number;
    void *data;
    uint32_t b_state; // BH_Uptodate, BH_Dirty
    struct list_head b_lru;
};
```

---

## 结论：构建可扩展的存储基石

块设备抽象层是自制操作系统**存储栈的基石**。  
通过统一接口和分区支持，  
我们实现了：
- **硬件无关**：文件系统无需关心 IDE/AHCI/NVMe
- **分区自动识别**：MBR/GPT 无缝支持
- **设备即文件**：`/dev/hda1`、`/dev/sdb2` 统一访问
- **易于扩展**：新增驱动只需实现 `block_operations`

此框架为后续实现 **ext2、tmpfs、swap** 奠定了坚实基础。  
真正的操作系统，始于对硬件的抽象与封装。

---

## 附录：关键接口速查

### 块设备注册
```c
struct block_device *block_register_device(
    const char *name, 
    uint8_t major,
    struct block_operations *ops,
    void *private_data);
```

### 通用 I/O
```c
int block_read(struct block_device *bdev, uint64_t sector, void *buffer, uint32_t count);
int block_write(struct block_device *bdev, uint64_t sector, const void *buffer, uint32_t count);
```

### 设备查找
```c
struct block_device *block_find_by_name(const char *name);
```

### 分区设备命名
- **MBR**：`hda1`, `hda2`, `hda3`, `hda4`（主分区/扩展）
- **GPT**：`sda1`, `sda2`, ...（按顺序）

> **注**：本文所有代码均为简化实现，实际使用需添加错误处理、边界检查等。
