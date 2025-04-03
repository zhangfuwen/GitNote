
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
超级块的大小固定为 **1024 字节**，其数据结构在 Linux 内核中定义为 `struct ext2_super_block`（位于 [这里](https://github.com/torvalds/linux/blob/a1b5bd45d4ee58af4f56e49497b8c3db96d8f8a3/fs/ext2/ext2.h#L411)）。以下是关键字段的说明（顺序不对，顺序看图）：

| **字段名**               | **类型** | **描述**                                           |
| --------------------- | ------ | ------------------------------------------------ |
| `s_inodes_count`      | __le32 | 文件系统中 inode 的总数。                                 |
| `s_blocks_count`      | __le32 | 文件系统中数据块的总数。                                     |
| `s_r_blocks_count`    | __le32 | 为超级用户保留的块数（通常为总块数的 5%）。                          |
| `s_free_blocks_count` | __le32 | 当前可用的空闲块数。                                       |
| `s_free_inodes_count` | __le32 | 当前可用的空闲 inode 数。                                 |
| `s_first_data_block`  | __le32 | 第一个数据块的位置（通常为 1，块 0 保留给引导扇区）。                    |
| `s_log_block_size`    | __le32 | 块大小的对数（以 2 为底），例如：`0` 表示 1024 字节，`1` 表示 2048 字节。 |
| `s_blocks_per_group`  | __le32 | 每个块组包含的数据块数。                                     |
| `s_inodes_per_group`  | __le32 | 每个块组包含的 inode 数。                                 |
| `s_magic`             | __le16 | 魔数（Magic Number），用于验证文件系统类型，ext2 的魔数为 `0xEF53`。  |
| `s_state`             | __le16 | 文件系统状态（如已挂载、需要检查等）。                              |
| `s_mnt_count`         | __le16 | 文件系统已挂载的次数。                                      |
| `s_max_mnt_count`     | __le16 | 文件系统允许的最大挂载次数，超过后会提示执行 `e2fsck`。                 |
|                       |        |                                                  |
下图是linux下的定义，代码在[这里](https://github.com/torvalds/linux/blob/a1b5bd45d4ee58af4f56e49497b8c3db96d8f8a3/fs/ext2/ext2.h#L411)：
![linux的super_block定义](assets/Pasted%20image%2020250403101559.png)
下图是osdev的截图，说的是一致的，但是注意，super block是sector 2和3：
![](assets/Pasted%20image%2020250403101402.png)
#### 2. 关键字段的计算与应用
- **块大小**：  
  块大小（以字节为单位）通过公式 `block_size = 1024 << s_log_block_size` 计算。例如，若 `s_log_block_size` 为 `2`，则块大小为 `4096` 字节。
- **块组数量**：  
  总块组数量由 `s_blocks_count / s_blocks_per_group` 确定。每个块组独立管理 inode 和数据块，提高文件系统的可靠性和性能。
- **魔数验证**：  
  通过 `s_magic` 字段判断文件系统类型。若魔数不为 `0xEF53`，可能表示文件系统损坏或非 ext2 格式。

### 二、超级块的备份机制
为防止主超级块损坏，ext2 会在每个块组的起始位置备份超级块。例如：
- **主超级块**：位于块组 0 的1024字节处，与块大小无关。
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

## inode

在 ext2 文件系统中，**inode（索引节点）**是存储文件或目录元数据的核心结构，每个文件或目录在创建时都会被分配一个唯一的 inode。它不仅包含文件的属性信息，还通过指针体系管理文件的数据存储，是文件系统实现高效访问和数据组织的关键。以下是其深度解析：

### 一、inode 的核心结构与字段
#### 1. 基础布局
inode 的大小通常为 **128 字节**（ext2）或 **256 字节**（ext4），其结构在 Linux 内核中定义为 `struct ext2_inode`（位于[这里](https://github.com/torvalds/linux/blob/a1b5bd45d4ee58af4f56e49497b8c3db96d8f8a3/fs/ext2/ext2.h#L290)）。以下是核心字段的说明（但顺序不对，不要做为顺序参考）：

| **字段名**               | **类型** | **描述**                                                                 |
|--------------------------|----------|--------------------------------------------------------------------------|
| `i_mode`                 | __le16   | 文件类型（如普通文件、目录、符号链接）和权限位（rwxrwxrwx）。                 |
| `i_uid`                  | __le16   | 文件所有者的用户 ID。                                                    |
| `i_gid`                  | __le16   | 文件所属组的组 ID。                                                      |
| `i_size`                 | __le32   | 文件的字节大小。                                                         |
| `i_atime`                | __le32   | 最后访问时间戳（精确到秒）。                                              |
| `i_mtime`                | __le32   | 最后修改时间戳（精确到秒）。                                              |
| `i_ctime`                | __le32   | inode 元数据最后变更时间戳（精确到秒）。                                   |
| `i_links_count`          | __le16   | 硬链接数量（文件被引用的次数）。                                          |
| `i_blocks`               | __le32   | 文件占用的块数（以 512 字节为单位）。                                     |
| `i_block[EXT2_N_BLOCKS]` | __le32   | 数据块指针数组，支持直接、间接和双重间接块。                              |

具体结构在[这里](https://github.com/torvalds/linux/blob/a1b5bd45d4ee58af4f56e49497b8c3db96d8f8a3/fs/ext2/ext2.h#L290)，截图如下：

![](assets/Pasted%20image%2020250403104504.png)

osdev上的说明：

![](assets/Pasted%20image%2020250403104757.png)

inode size我的是256，版本是1。first inode是11, 所以前10个也是保留的。
#### 2. 数据块指针体系
- **直接块（Direct Blocks）**：  
  前 **12 个指针**（`i_block[0..11]`）直接指向数据块，适合快速访问小文件（如 12 个块 × 4KB 块大小 = 48KB）。
- **间接块（Indirect Blocks）**：  
  `i_block[12]` 指向一个块，该块存储 **256 个数据块指针**（单间接块），支持更大的文件（如 256 × 4KB = 1MB）。
- **双重间接块（Double Indirect Blocks）**：  
  `i_block[13]` 指向一个块，该块存储 **256 个单间接块指针**，每个单间接块又指向 256 个数据块（总容量为 256 × 256 × 4KB = 256MB）。
- **三重间接块（Triple Indirect Blocks）**：  
  `i_block[14]` 指向一个块，该块存储 **256 个双重间接块指针**，理论上支持的文件大小可达 **2^32 块**（受限于文件系统总块数）。

**示例计算**：  
若块大小为 4KB，一个文件使用 12 个直接块 + 1 个单间接块 + 1 个双重间接块，其总大小为：  
`12×4KB + 256×4KB + 256×256×4KB = 48KB + 1MB + 256MB = 257.048MB`。

### 二、inode 的功能与工作原理
#### 1. 文件类型与权限控制
- **文件类型**：  
  通过 `i_mode` 的高 4 位（`S_IFMT` 掩码）判断文件类型，如：
  - `S_IFREG`（0x8000）：普通文件。
  - `S_IFDIR`（0x4000）：目录。
  - `S_IFLNK`（0xA000）：符号链接。
- **权限位**：  
  `i_mode` 的低 12 位（`S_IRWXU/S_IRWXG/S_IRWXO`）分别控制所有者、所属组、其他用户的读（r）、写（w）、执行（x）权限。

#### 2. 数据存储与访问
- **小文件优化**：  
  对于小于等于 12 个块的文件，数据直接存储在直接块中，无需额外寻址开销。
- **大文件支持**：  
  通过多级间接块指针，ext2 理论上支持单个文件最大为 **16TB**（块大小 4KB 时）。
- **符号链接优化**：  
  若符号链接的目标路径长度 ≤ 60 字节，路径直接存储在 inode 的 `i_block` 中，无需额外数据块。

#### 3. 时间戳与元数据管理
- **访问时间（atime）**：  
  每次读取文件时更新，可通过 `mount` 选项 `noatime` 禁用以提升性能。
- **修改时间（mtime）**：  
  文件内容变更时更新。
- **变更时间（ctime）**：  
  inode 元数据（如权限、所有者）变更时更新。

### 三、inode 的存储与管理
#### 1. 物理布局
- **inode 表**：  
  每个块组包含一个 inode 表，存储该组内所有 inode 的二进制数据。例如，一个块组包含 8192 个 inode，每个 inode 占 128 字节，则 inode 表大小为 `8192 × 128B = 1MB`。
- **inode 位图**：  
  每个块组有一个 inode 位图，用每一位表示对应的 inode 是否被占用（0 表示空闲，1 表示已分配）。

#### 2. 分配与回收
- **分配流程**：  
  1. 扫描 inode 位图，找到第一个空闲位。
  2. 标记该位为已使用，初始化 inode 元数据（如权限、所有者）。
  3. 若为目录，创建 `.`（当前目录）和 `..`（父目录）的目录项。
- **回收流程**：  
  1. 减少 `i_links_count`，若为 0，则标记 inode 位图对应位为空闲。
  2. 释放所有关联的数据块（通过指针遍历直接、间接块）。
  3. 更新超级块中的空闲 inode 计数。

#### 3. 数量限制
- **inode 总数**：  
  格式化时由 `mke2fs` 根据分区大小和块组参数自动计算，公式为：  
  `inode_count = (total_blocks / blocks_per_group) × inodes_per_group`。
- **inode 耗尽问题**：  
  若文件系统包含大量小文件（如日志文件），可能耗尽 inode 而不是磁盘空间。可通过 `df -i` 查看 inode 使用情况。

### 四、inode 的查看与维护
#### 1. 命令行工具
- **`ls -i`**：  
  显示文件或目录的 inode 编号：
  ```bash
  $ ls -i /etc/passwd
  262145 /etc/passwd
  ```
- **`stat`**：  
  查看 inode 元数据：
  ```bash
  $ stat /etc/passwd
  File: /etc/passwd
  Size: 2050       	Blocks: 8          IO Block: 4096   regular file
  Device: 801h/2049d	Inode: 262145      Links: 1
  Access: (0644/-rw-r--r--)  Uid: (    0/    root)   Gid: (    0/    root)
  Access: 2025-04-03 10:41:23.000000000 +0800
  Modify: 2025-03-25 15:32:47.000000000 +0800
  Change: 2025-03-25 15:32:47.000000000 +0800
  Birth: -
  ```
- **`debugfs`**：  
  直接访问 inode 结构（需 root 权限）：
  ```bash
  $ sudo debugfs -R 'stat <inode_number>' /dev/sda1
  Inode: 262145   Type: regular    Mode:  0644   Flags: 0x80000
  Generation: 0   Version: 0x00000000
  User: 0   Group: 0   Size: 2050
  ...
  ```

#### 2. 修复工具
- **`e2fsck`**：  
  检查并修复 inode 错误（需卸载文件系统）：
  ```bash
  $ sudo umount /dev/sda1
  $ sudo e2fsck -f /dev/sda1
  ```
- **`debugfs`**：  
  手动修复损坏的 inode（危险操作，需谨慎）：
  ```bash
  $ sudo debugfs /dev/sda1
  debugfs:  clri <inode_number>  # 清除 inode 内容
  debugfs:  write <file> <inode_number>  # 从文件恢复 inode
  ```

### 五、inode 的高级特性与优化
#### 1. 硬链接与软链接
- **硬链接**：  
  多个文件名指向同一个 inode，`i_links_count` 增加。删除其中一个文件名不会影响数据，直到 `i_links_count` 为 0。
- **软链接**：  
  独立 inode，存储目标路径，不增加 `i_links_count`。目标文件删除后，软链接失效。

#### 2. 扩展属性（XATTR）
- **功能**：  
  在 inode 中存储额外元数据（如文件分类、加密标记），通过 `setfattr`/`getfattr` 操作。
- **实现**：  
  ext2 通过 `i_block` 存储扩展属性数据块，需在格式化时启用（`mke2fs -O extent`）。

#### 3. 性能优化
- **预分配 inode**：  
  `mke2fs -N <number>` 可手动指定 inode 数量，避免自动计算导致的浪费。
- **碎片整理**：  
  `e4defrag` 工具可优化 inode 和数据块的物理布局，提升访问速度。

### 六、常见问题与解决方案
| **问题**               | **现象**                              | **解决方案**                          |
|------------------------|---------------------------------------|---------------------------------------|
| inode 耗尽             | `df -i` 显示 100%，无法创建新文件     | 清理小文件或扩容分区                  |
| inode 元数据损坏       | 文件无法访问，`stat` 报错             | `e2fsck -f` 修复或从备份恢复          |
| 间接块指针错误         | 文件读取失败，`debugfs` 提示无效块号  | 手动修复指针或使用 `e2fsck` 自动修复  |
| 硬链接计数异常         | `ls -l` 显示 `??` 链接数              | `e2fsck -l` 重建链接计数              |

### 七、总结
inode 是 ext2 文件系统的“灵魂”，其设计兼顾了小文件的快速访问和大文件的高效存储。通过理解 inode 的结构、指针体系和管理工具，管理员可深入排查文件系统问题，优化性能，并在数据恢复中发挥关键作用。在实际操作中，建议结合 `ls`、`stat`、`debugfs` 和 `e2fsck` 等工具，全面掌握 inode 的状态与行为。