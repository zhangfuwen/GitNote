# 存储系统-高级篇：为自制 OS 实现现代存储特性

> **“现代企业级存储需要快照、压缩、加密、分布式等高级特性。  
> 本文将解析 CoW、透明压缩、文件加密的核心机制，  
> 并为自制 OS 提供可实现的简化方案。”**

## 引言：高级存储特性的必要性

在自制操作系统中，基础文件系统（如 ext2）已能满足简单需求。  
但**现代应用场景**要求更多高级特性：
- **数据保护**：快照防止误删除
- **空间节省**：透明压缩减少存储成本
- **安全合规**：文件级加密保护敏感数据
- **横向扩展**：分布式文件系统支持 PB 级存储
- **新型硬件**：持久内存（PMEM）需要专用文件系统

本文将为自制 OS 设计**可实现的简化方案**，  
涵盖 CoW 快照、透明压缩、文件加密等核心特性。

---

## 第一章：写时复制（CoW）与快照

### 1.1 CoW 核心思想

#### 问题：
- **传统快照**：复制整个文件系统 → 耗时耗空间
- **写入放大**：修改小文件需复制大文件

#### 解决方案：
- **写时复制**（Copy-on-Write）：共享数据块，写时才复制
- **快照**：记录文件系统元数据状态

#### CoW 优势：
- **空间高效**：快照仅存储差异
- **时间高效**：快照创建 O(1)
- **一致性**：快照保证原子性

### 1.2 CoW 文件系统设计

#### 元数据结构：
```c
// cowfs.h
struct cowfs_inode {
    uint64_t i_ino;
    uint32_t i_gen;         // 生成号（快照标识）
    uint64_t i_data_blocks; // 数据块指针（CoW 树）
    // ... 其他字段
};

struct cowfs_block_ref {
    uint64_t block_id;      // 块 ID
    uint32_t ref_count;     // 引用计数
    uint32_t flags;         // COW_BLOCK_SHARED
};
```

#### 写时复制流程：
1. **读取块**：增加引用计数
2. **写入块**：
   - 检查引用计数 > 1 → 复制新块
   - 更新块指针
   - 减少旧块引用计数

```c
// cowfs.c
static int cowfs_write_block(struct cowfs_inode *inode, 
                            uint64_t block_index, 
                            const void *data) {
    // 1. 获取当前块引用
    struct cowfs_block_ref *old_ref = 
        cowfs_get_block_ref(inode, block_index);
    
    // 2. 检查是否共享
    if (old_ref->ref_count > 1) {
        // 3. 分配新块
        uint64_t new_block = cowfs_alloc_block();
        cowfs_write_block_data(new_block, data);
        
        // 4. 创建新引用
        struct cowfs_block_ref *new_ref = 
            cowfs_create_ref(new_block);
        new_ref->ref_count = 1;
        
        // 5. 更新 inode
        cowfs_set_block_ref(inode, block_index, new_ref);
        
        // 6. 减少旧引用
        cowfs_put_block_ref(old_ref);
    } else {
        // 直接覆盖
        cowfs_write_block_data(old_ref->block_id, data);
    }
    return 0;
}
```

### 1.3 快照实现

#### 快照创建：
```c
// cowfs.c
int cowfs_create_snapshot(const char *name) {
    // 1. 分配快照 ID
    uint32_t snap_id = cowfs_next_snap_id();
    
    // 2. 复制根 inode（仅元数据）
    struct cowfs_inode *root = cowfs_get_root_inode();
    struct cowfs_inode *snap_root = cowfs_copy_inode(root);
    snap_root->i_gen = snap_id;
    
    // 3. 保存快照元数据
    cowfs_save_snapshot(snap_id, snap_root, name);
    return 0;
}
```

#### 快照挂载：
```c
// vfs/cowfs.c
struct vfs_super *cowfs_mount_snapshot(const char *dev, uint32_t snap_id) {
    // 1. 读取快照元数据
    struct cowfs_snapshot *snap = cowfs_load_snapshot(snap_id);
    
    // 2. 创建只读超级块
    struct vfs_super *sb = kmalloc(sizeof(struct vfs_super));
    sb->s_root = d_obtain_root(snap->root_inode);
    sb->s_flags |= MS_RDONLY; // 快照只读
    
    return sb;
}
```

---

## 第二章：透明压缩

### 2.1 透明压缩设计

#### 核心思想：
- **自动压缩/解压**：对应用透明
- **按页压缩**：4KB 页独立压缩
- **压缩算法**：zstd（高压缩比+快速）

#### 元数据结构：
```c
// compressfs.h
struct compressfs_inode {
    uint64_t i_compressed_blocks; // 压缩块指针
    uint32_t i_compression_alg;   // 压缩算法（ZSTD=1）
};

struct compressfs_block_header {
    uint32_t magic;     // 0xC0MP
    uint32_t orig_size; // 原始大小
    uint32_t comp_size; // 压缩大小
    // 压缩数据...
};
```

### 2.2 读写流程

#### 写入（压缩）：
```c
// compressfs.c
static int compressfs_write_page(struct vfs_inode *inode, 
                                uint64_t index, 
                                const void *data) {
    // 1. 压缩数据
    void *comp_data = kmalloc(PAGE_SIZE);
    size_t comp_size = zstd_compress(comp_data, PAGE_SIZE, data, PAGE_SIZE);
    
    // 2. 构建块头
    struct compressfs_block_header *hdr = kmalloc(sizeof(*hdr) + comp_size);
    hdr->magic = 0xC0MP;
    hdr->orig_size = PAGE_SIZE;
    hdr->comp_size = comp_size;
    memcpy(hdr + 1, comp_data, comp_size);
    
    // 3. 写入磁盘
    uint64_t block = compressfs_alloc_block();
    block_write(inode->i_sb->s_bdev, block, hdr, 1);
    
    kfree(hdr);
    kfree(comp_data);
    return 0;
}
```

#### 读取（解压）：
```c
static int compressfs_read_page(struct vfs_inode *inode, 
                               uint64_t index, 
                               void *data) {
    // 1. 读取压缩块
    struct compressfs_block_header *hdr = kmalloc(PAGE_SIZE);
    block_read(inode->i_sb->s_bdev, block_addr, hdr, 1);
    
    // 2. 验证魔数
    if (hdr->magic != 0xC0MP) return -1;
    
    // 3. 解压数据
    zstd_decompress(data, hdr->orig_size, hdr + 1, hdr->comp_size);
    
    kfree(hdr);
    return 0;
}
```

### 2.3 压缩策略

#### 何时压缩？
- **小文件**：>50% 压缩率才存储
- **大文件**：分段压缩（每 4KB 独立）
- **随机访问**：解压单个页，不影响其他页

#### 压缩算法选择：
| 算法 | 压缩比 | 速度 | 适用场景 |
|------|--------|------|----------|
| **zstd** | 高 | 快 | 通用 |
| **lz4** | 低 | 极快 | 实时 |
| **gzip** | 中 | 慢 | 归档 |

> 💡 **自制 OS 建议**：从 zstd 开始，平衡压缩比与速度

---

## 第三章：文件级加密

### 3.1 fscrypt 设计思想

#### 核心原则：
- **文件级加密**：每个文件独立密钥
- **元数据不加密**：文件名、大小可见
- **密钥层次**：
  - **Master Key**：用户主密钥
  - **File Key**：文件密钥（由 Master Key 加密存储）

#### 加密流程：
1. **创建文件**：
   - 生成随机 File Key
   - 用 Master Key 加密 File Key
   - 存储加密后的 File Key 到 inode
2. **读写文件**：
   - 用 Master Key 解密 File Key
   - 用 File Key 加解密数据

### 3.2 简化加密实现

#### inode 扩展：
```c
// encryptfs.h
struct encryptfs_inode {
    uint8_t i_nonce[16];        // 随机 nonce
    uint8_t i_enc_file_key[32]; // 加密的文件密钥
    uint32_t i_key_id;          // 密钥 ID
};
```

#### 加密写入：
```c
// encryptfs.c
static int encryptfs_write_page(struct vfs_inode *inode, 
                               uint64_t index, 
                               const void *data) {
    struct encryptfs_inode *ei = inode->i_private;
    
    // 1. 获取文件密钥（解密）
    uint8_t file_key[32];
    fscrypt_decrypt_key(ei->i_enc_file_key, file_key, master_key);
    
    // 2. 生成随机 IV
    uint8_t iv[16];
    memcpy(iv, ei->i_nonce, 12);
    memcpy(iv + 12, &index, 4);
    
    // 3. AES-XTS 加密
    uint8_t *encrypted = kmalloc(PAGE_SIZE);
    aes_xts_encrypt(encrypted, data, PAGE_SIZE, file_key, iv);
    
    // 4. 写入磁盘
    // ... block_write
    
    kfree(encrypted);
    return 0;
}
```

#### 密钥管理：
```c
// encryptfs.c
int sys_add_encryption_key(const char *key_descriptor, 
                          const uint8_t *raw_key, 
                          uint32_t key_size) {
    // 1. 验证密钥
    if (key_size != 32) return -1;
    
    // 2. 保存到内核密钥环
    struct fscrypt_key *key = kmalloc(sizeof(*key));
    memcpy(key->raw, raw_key, key_size);
    key->id = fscrypt_key_id(key_descriptor);
    
    list_add_tail(&key->list, &fscrypt_keyring);
    return 0;
}
```

---

## 第四章：分布式文件系统

### 4.1 分布式存储核心挑战

#### 三大问题：
- **数据分布**：如何分片存储到多节点？
- **一致性**：如何保证多副本一致？
- **容错**：节点宕机如何恢复？

#### 解决方案对比：
| 系统 | 数据分布 | 一致性 | 容错 |
|------|----------|--------|------|
| **Ceph** | CRUSH 算法 | 强一致 | 多副本/纠删码 |
| **GlusterFS** | 弹性哈希 | 最终一致 | AFR（自动文件复制） |
| **HDFS** | 主从架构 | 强一致 | 三副本 |

### 4.2 简化分布式实现（GlusterFS 风格）

#### 客户端架构：
```c
// distfs.h
struct distfs_super {
    struct list_head servers; // 存储服务器列表
    uint32_t replica_count;   // 副本数
};

struct distfs_server {
    char hostname[64];
    int port;
    int sockfd;
};
```

#### 文件写入流程：
1. **分片**：文件按 128KB 分片
2. **哈希**：分片 ID → 服务器列表
3. **并行写**：同时写多个副本

```c
// distfs.c
static int distfs_write_page(struct vfs_inode *inode, 
                            uint64_t index, 
                            const void *data) {
    struct distfs_super *ds = inode->i_sb->s_fs_info;
    uint64_t chunk_id = (inode->i_ino << 32) | index;
    
    // 1. 计算服务器列表
    struct distfs_server *servers[ds->replica_count];
    distfs_get_servers(chunk_id, servers, ds->replica_count);
    
    // 2. 并行写入
    for (int i = 0; i < ds->replica_count; i++) {
        distfs_send_write(servers[i], chunk_id, data, PAGE_SIZE);
    }
    
    return 0;
}
```

#### 网络协议（简化）：
```c
// distfs_wire.h
struct distfs_request {
    uint32_t magic;     // 0xD157
    uint32_t op;        // OP_WRITE=1, OP_READ=2
    uint64_t chunk_id;
    uint32_t size;
    // 数据...
};
```

---

## 第五章：持久内存（PMEM）文件系统

### 5.1 PMEM 硬件特性

#### 关键特性：
- **字节寻址**：像内存一样访问
- **持久性**：断电不丢数据
- **低延迟**：比 SSD 快 1000 倍

#### 挑战：
- **缓存刷新**：需 `clflush` 确保持久化
- **原子性**：写操作需 8 字节对齐

### 5.2 PMEM 文件系统设计

#### 元数据布局：
```c
// pmemfs.h
struct pmemfs_super {
    uint64_t magic;         // 0x504D454D46530001
    uint64_t root_inode_offset;
    uint64_t free_list_offset;
    // ... 其他字段
} __attribute__((aligned(64)));

struct pmemfs_inode {
    uint64_t i_ino;
    uint32_t i_mode;
    uint64_t i_size;
    uint64_t i_data[15];    // 直接块指针（PMEM 偏移）
} __attribute__((aligned(64)));
```

#### 持久化写入：
```c
// pmemfs.c
static void pmem_persist(void *addr, size_t len) {
    char *end = (char*)addr + len;
    for (char *p = (char*)addr; p < end; p += 64) {
        __builtin_ia32_clflush(p);
    }
    __builtin_ia32_sfence(); // 内存屏障
}

static int pmemfs_write_block(struct pmemfs_inode *inode, 
                             uint64_t block_index, 
                             const void *data) {
    // 1. 计算 PMEM 地址
    void *pmem_addr = pmem_region + inode->i_data[block_index];
    
    // 2. 复制数据
    memcpy(pmem_addr, data, 4096);
    
    // 3. 持久化
    pmem_persist(pmem_addr, 4096);
    return 0;
}
```

### 5.3 无日志设计

#### 优势：
- **零拷贝**：直接写 PMEM
- **低延迟**：无日志开销
- **简单**：无需日志恢复

#### 保证一致性：
- **原子更新**：8 字节对齐的指针更新
- **校验和**：检测静默损坏

---

## 第六章：高级特性整合

### 6.1 特性组合

#### 典型企业配置：
- **CoW + 快照**：数据保护
- **透明压缩**：节省空间
- **文件加密**：安全合规
- **分布式**：横向扩展

#### 性能权衡：
| 特性 | CPU 开销 | I/O 开销 | 空间节省 |
|------|----------|----------|----------|
| **CoW** | 低 | 中 | 高（快照） |
| **压缩** | 高 | 低 | 高 |
| **加密** | 中 | 低 | 无 |
| **分布式** | 低 | 高 | 无（多副本） |

### 6.2 自制 OS 实现建议

#### 分阶段实现：
1. **第一阶段**：CoW 快照（最实用）
2. **第二阶段**：透明压缩（zstd）
3. **第三阶段**：文件加密（AES-XTS）
4. **第四阶段**：分布式客户端（可选）

#### 硬件要求：
- **CoW/压缩/加密**：普通 PC 即可
- **分布式**：需多台机器或虚拟机
- **PMEM**：需 Intel DCPMM 硬件

---

## 结论：迈向企业级存储

高级存储特性是自制操作系统**走向实用的关键一步**。  
通过 CoW、压缩、加密的协同，  
我们构建了：
- **数据保护**：快照防止误操作
- **空间优化**：透明压缩节省成本
- **安全保障**：文件级加密满足合规
- **未来扩展**：分布式与 PMEM 支持

这些特性虽复杂，但**分阶段实现**完全可行。  
真正的企业级存储，始于对高级特性的深刻理解与务实实现。

---

## 附录：关键实现要点

### CoW 快照
- **引用计数**：精确跟踪块共享
- **生成号**：区分快照版本
- **只读挂载**：快照不可修改

### 透明压缩
- **按页压缩**：4KB 独立压缩
- **压缩率检测**：低压缩率回退
- **zstd 库**：集成开源压缩库

### 文件加密
- **密钥层次**：Master Key → File Key
- **AES-XTS**：标准磁盘加密模式
- **密钥环**：安全存储密钥

### 分布式
- **分片哈希**：确定性数据分布
- **多副本**：简单容错
- **网络协议**：轻量级二进制协议

> **注**：本文所有代码均为简化实现，实际使用需添加错误处理、并发控制、安全审计等。