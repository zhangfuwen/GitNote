
# qemu的xxx.img格式磁盘

其实就是一个普通的二进制文件，里面没有任何格式，它就是dd的结果。生成一个这样的磁盘只需要

```bash
# 创建空的磁盘镜像文件  
dd if=/dev/zero of=$disk_image bs=1M count=100
```

# ext2文件系统格式

## 制作方法

```bash

# 格式化为ext2文件系统  
mkfs.ext2 $disk_image  
  
# 创建临时挂载点  
tmp_mount=$(mktemp -d)  
  
# 挂载磁盘镜像  
sudo mount -o loop $disk_image $tmp_mount  
  
# 检查rootfs/binary目录是否存在  
if [ -d "rootfs/binary" ]; then  
    # 复制文件到磁盘镜像  
    sudo cp -r rootfs/binary/* $tmp_mount/  
else  
    echo "错误：rootfs/binary目录不存在"  
fi  
  
# 卸载磁盘镜像  
sudo umount $tmp_mount
```

## 查看文件内容

```bash

sudo mount -o loop $disk_image /mnt 
ls /mnt 
sudo umount /mnt

```


## 超级块

在 ext2 文件系统中，**超级块（Super Block）**是存储文件系统全局元数据的核心结构，包含文件系统的布局、容量、状态等关键信息。以下是其格式的详细解析：

### 一、超级块的结构与核心字段
#### 1. 基本结构
超级块的大小固定为 **1024 字节**，其数据结构在 Linux 内核中定义为 `struct ext2_super_block`（位于 `include/linux/ext2_fs.h`）。以下是关键字段的说明：

| **字段名**               | **类型** | **描述**                                                                 |
|--------------------------|----------|--------------------------------------------------------------------------|
| `s_inodes_count`         | __le32   | 文件系统中 inode 的总数。                                               |
| `s_blocks_count`         | __le32   | 文件系统中数据块的总数。                                               |
| `s_r_blocks_count`       | __le32   | 为超级用户保留的块数（通常为总块数的 5%）。                            |
| `s_free_blocks_count`    | __le32   | 当前可用的空闲块数。                                                   |
| `s_free_inodes_count`    | __le32   | 当前可用的空闲 inode 数。                                               |
| `s_first_data_block`     | __le32   | 第一个数据块的位置（通常为 1，块 0 保留给引导扇区）。                   |
| `s_log_block_size`       | __le32   | 块大小的对数（以 2 为底），例如：`0` 表示 1024 字节，`1` 表示 2048 字节。 |
| `s_blocks_per_group`     | __le32   | 每个块组包含的数据块数。                                               |
| `s_inodes_per_group`     | __le32   | 每个块组包含的 inode 数。                                               |
| `s_magic`                | __le16   | 魔数（Magic Number），用于验证文件系统类型，ext2 的魔数为 `0xEF53`。   |
| `s_state`                | __le16   | 文件系统状态（如已挂载、需要检查等）。                                  |
| `s_mnt_count`            | __le16   | 文件系统已挂载的次数。                                                 |
| `s_max_mnt_count`        | __le16   | 文件系统允许的最大挂载次数，超过后会提示执行 `e2fsck`。               |

#### 2. 关键字段的计算与应用
- **块大小**：  
  块大小（以字节为单位）通过公式 `block_size = 1024 << s_log_block_size` 计算。例如，若 `s_log_block_size` 为 `2`，则块大小为 `4096` 字节。
- **块组数量**：  
  总块组数量由 `s_blocks_count / s_blocks_per_group` 确定。每个块组独立管理 inode 和数据块，提高文件系统的可靠性和性能。
- **魔数验证**：  
  通过 `s_magic` 字段判断文件系统类型。若魔数不为 `0xEF53`，可能表示文件系统损坏或非 ext2 格式。

### 二、超级块的备份机制
为防止主超级块损坏，ext2 会在每个块组的起始位置备份超级块。例如：
- **主超级块**：位于块组 0 的块 1（若块大小为 1024 字节）。
- **备份超级块**：分布在块组 1、2、3 等的起始块（具体位置由 `mke2fs` 工具在格式化时确定）。

**恢复方法**：  
使用 `e2fsck -b <备份块号> /dev/sda1` 命令指定备份超级块进行文件系统修复。例如，`e2fsck -b 8193 /dev/sda1` 表示使用块号为 8193 的备份超级块。

### 三、超级块的实际查看与解析
#### 1. 使用 `dumpe2fs` 工具
通过 `dumpe2fs -h /dev/sda1` 命令可查看超级块的详细信息：
```bash
# dumpe2fs -h /dev/sda1
Filesystem volume name:   <none>
Last mounted on:          /mnt
Filesystem UUID:          5b6740c3-...
Filesystem magic number:  0xEF53
Filesystem revision #:    1 (dynamic)
Filesystem features:      ext_attr resize_inode dir_index filetype sparse_super
Block size:               4096
Blocks per group:         32768
Inodes per group:         8192
Reserved block count:     16384
Free blocks:              1048576
Free inodes:              81920
First data block:         0
```

#### 2. 二进制数据解析
使用 `hexdump` 或 `dd` 工具可直接读取超级块的二进制内容：
```bash
# hexdump -n 1024 -s 1024 /dev/sda1 | less
00000400  20 63 00 00 00 8c 01 00  cc 13 00 00 67 79 01 00  | c...........gy..|
00000410  15 63 00 00 01 00 00 00  00 00 00 00 00 00 00 00  |.c..............|
...
00000430  40 f0 e3 66 00 00 ff ff  53 ef 01 00 01 00 00 00  |@..f....S.......|
```
- **魔数**：`0xEF53` 位于偏移量 `0x438` 处（小端序存储为 `53 ef`）。
- **块大小**：`s_log_block_size` 字段为 `0x02`，对应块大小为 `4096` 字节。

### 四、超级块的版本差异与扩展
#### 1. 版本演进
- **ext2**：基础版本，无日志功能。
- **ext3**：在 ext2 基础上增加日志功能，魔数仍为 `0xEF53`，但通过 `s_feature_incompat` 字段标记日志特性。
- **ext4**：兼容 ext2/ext3，支持更大的文件系统和 inode 扩展（如 256 字节 inode）。

#### 2. 扩展字段
- **特性标志**：  
  `s_feature_compat`、`s_feature_ro_compat`、`s_feature_incompat` 字段记录文件系统支持的特性（如扩展属性、目录索引等）。
- **UUID 与标签**：  
  `s_uuid` 和 `s_volume_name` 字段用于唯一标识文件系统和设置卷标。

### 五、常见问题与维护
1. **超级块损坏**：  
   - 现象：文件系统无法挂载，提示 `Invalid superblock`。
   - 解决方案：使用备份超级块恢复（如 `e2fsck -b 8193 /dev/sda1`）。

2. **文件系统一致性检查**：  
   - 命令：`e2fsck -f /dev/sda1`。
   - 作用：检查并修复超级块、块组描述符、inode 表等元数据错误。

3. **调整文件系统参数**：  
   - 使用 `tune2fs` 工具修改超级块参数（如最大挂载次数、保留块比例等）：
     ```bash
     # tune2fs -c 0 /dev/sda1  # 关闭自动检查
     # tune2fs -m 10 /dev/sda1 # 将保留块比例调整为 10%
     ```

### 六、总结
超级块是 ext2 文件系统的“心脏”，存储了文件系统的全局信息。通过 `dumpe2fs`、`e2fsck` 等工具，管理员可查看、维护和修复超级块，确保文件系统的稳定性。理解超级块的结构和备份机制，有助于深入排查文件系统故障并优化性能。