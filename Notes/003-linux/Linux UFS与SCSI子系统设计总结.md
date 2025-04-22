好的！我将用通俗易懂的方式为你总结 Linux 内核中 **UFS（Universal Flash Storage）** 和 **SCSI（Small Computer System Interface）** 子系统的设计逻辑和协作关系。这两个子系统共同管理现代存储设备（如固态硬盘、移动设备存储等）。

---

## 一、核心概念
### 1. **SCSI 子系统**
- **作用**：SCSI 是一种广泛使用的存储协议标准，用于管理磁盘、SSD、光驱等设备。Linux 的 SCSI 子系统负责将通用块设备请求转换为具体的 SCSI 命令。
- **分层设计**：
  - **高层驱动**：如 `sd`（SCSI Disk）驱动，管理磁盘设备。
  - **中间层**：处理 SCSI 命令队列、错误恢复、设备发现等。
  - **低层驱动**：适配具体硬件（如 SAS、SATA、UFS 等）。

### 2. **UFS 子系统**
- **作用**：UFS 是面向移动设备和嵌入式系统的高速闪存协议（如手机存储）。Linux 的 UFS 子系统通过 SCSI 命令与硬件交互。
- **关键组件**：
  - **UFS 主机控制器驱动**：管理 UFS 硬件控制器（如 `ufs-hcd`）。
  - **UFS 核心层**：处理协议逻辑（如命令封装、错误处理）。
  - **SCSI 传输层**：将 UFS 操作映射到 SCSI 命令。

---

## 二、协作关系
### 1. UFS 作为 SCSI 的“客户端”
- UFS 设备在 Linux 中被抽象为 **SCSI 设备**，通过 SCSI 中间层与内核其他部分交互。
- **数据流**：
  ```
  用户层（读写文件）→ 内核 VFS → 块层（Block Layer）→ SCSI 子系统 → UFS 子系统 → UFS 硬件
  ```

### 2. SCSI 命令到 UFS 的转换
- **SCSI 命令**（如读/写）会被 UFS 子系统转换为 **UFS Protocol Information Units (UPIU)**。
- **示例**：用户发起一个读操作：
  1. SCSI 中间层生成 `READ_10` 命令。
  2. UFS 核心层将 `READ_10` 封装为 UPIU 格式。
  3. UFS 主机控制器通过硬件接口发送 UPIU 到 UFS 设备。

---

## 三、关键设计细节
### 1. **SCSI 中间层（Mid-Layer）**
- **功能**：
  - 管理 SCSI 命令队列（合并、排序、超时处理）。
  - 设备发现和热插拔（通过 `scsi_scan` 模块）。
  - 错误恢复（如自动重试、重置设备）。
- **数据结构**：
  - `struct scsi_cmnd`：表示一个 SCSI 命令。
  - `struct Scsi_Host`：表示一个 SCSI 主机适配器。

### 2. **UFS 核心层**
- **功能**：
  - 封装 UPIU（命令、数据、响应）。
  - 管理 UFS 设备的电源状态（如休眠、激活）。
  - 处理错误和异常（如 CRC 校验失败）。
- **关键模块**：
  - `ufshcd.c`：UFS 主机控制器驱动核心。
  - `ufs_quirks.c`：处理不同厂商的硬件兼容性问题。

### 3. **内存管理**
- **SCSI 和 UFS 使用 DMA**：通过 `dma_alloc_coherent()` 分配内存，实现高速数据传输。
- **Scatter-Gather 列表**：支持分散/聚集 I/O，提升大块数据传输效率。

---

## 四、代码结构（Linux 内核）
### 1. **SCSI 子系统**
- 目录：`drivers/scsi/`
  - `sd.c`：SCSI 磁盘驱动。
  - `scsi_debug.c`：调试工具。
  - `libsas/`：SAS（Serial Attached SCSI）驱动。

### 2. **UFS 子系统**
- 目录：`drivers/ufs/`
  - `ufs-hcd.c`：主机控制器驱动。
  - `ufs-mediatek.c`：某厂商的 UFS 控制器实现（如联发科）。
  - `ufs.h`：UFS 协议定义和数据结构。

---

## 五、调试与工具
### 1. **常用工具**
- `lsscsi`：列出所有 SCSI 设备。
- `sg_utils`：发送原始 SCSI 命令。
- `ufs-utils`：UFS 专用调试工具。

### 2. **内核日志**
- 查看 `dmesg` 输出，过滤关键字 `scsi` 或 `ufs`。
- 示例调试命令：
  ```bash
  dmesg | grep -E "scsi|ufs"
  ```

### 3. **性能调优**
- **调整队列深度**：通过 `sysfs` 修改 `/sys/block/sdX/queue/nr_requests`。
- **电源管理**：禁用 UFS 的自动休眠（通过 `ufshcd` 参数）。

---

## 六、总结
- **SCSI 子系统**：是 Linux 存储栈的通用抽象层，支持多种硬件。
- **UFS 子系统**：通过 SCSI 命令与硬件交互，专注于闪存特性优化。
- **协作流程**：用户请求 → 块层 → SCSI 中间层 → UFS 驱动 → 硬件。

通过这种分层设计，Linux 实现了存储协议的统一管理和硬件多样性支持。

---

如果你有具体问题（如代码逻辑、某个模块的细节），可以进一步讨论！



![](assets/Pasted%20image%2020250422155833.png)

好的！我将从 **高层驱动**、**中间层**、**底层驱动** 以及它们之间的接口设计角度，详细解释 Linux 中 SCSI 和 UFS 子系统的协作关系。这里会结合代码结构、数据流和实际接口示例。

---

## 一、整体分层架构
在 Linux 存储子系统中，各层的分工如下：

```
用户层（User Space）
   ↓ 系统调用（read/write）
虚拟文件系统（VFS）
   ↓ 文件操作（struct file_operations）
块层（Block Layer）
   ↓ 通用块请求（struct bio）
SCSI 子系统高层驱动（如 sd.c）
   ↓ SCSI 命令（struct scsi_cmnd）
SCSI 中间层（SCSI Mid-Layer）
   ↓ SCSI 主机适配器接口（struct Scsi_Host）
SCSI 底层驱动（如 SAS/SATA/UFS 驱动）
   ↓ 硬件协议（如 UFS UPIU、SATA FIS）
硬件设备（UFS 控制器、SSD 等）
```

---

## 二、各层功能及接口详解

### 1. **高层驱动（SCSI Upper Layer）**
**代表模块**：`sd.c`（SCSI Disk 驱动）  
**作用**：将块层的通用块设备请求（`struct bio`）转换为 SCSI 命令（`struct scsi_cmnd`）。

#### 关键接口与数据结构：
- **块层接口**：
  - 通过 `struct gendisk` 和 `struct request_queue` 管理块设备。
  - 处理块请求的回调函数：`.request_fn` 或更现代的 `.queue_rq`（多队列驱动）。
  ```c
  // 示例：块层请求处理（drivers/scsi/sd.c）
  static const struct blk_mq_ops sd_mq_ops = {
      .queue_rq       = sd_queue_rq,   // 处理块层请求
      .complete       = sd_complete,   // 请求完成回调
  };
  ```

- **SCSI 命令生成**：
  - 将 `struct bio` 转换为 `struct scsi_cmnd`，填充 SCSI 命令（如 `READ_10`, `WRITE_10`）。
  ```c
  // 生成 SCSI 读命令（drivers/scsi/scsi_lib.c）
  void scsi_init_io(struct scsi_cmnd *cmd) {
      struct request *req = cmd->request;
      cmd->cmnd[0] = READ_10;  // 操作码
      // 填充 LBA、长度等参数
  }
  ```

---

### 2. **SCSI 中间层（SCSI Mid-Layer）**
**核心模块**：`scsi_lib.c`, `scsi_error.c`  
**作用**：管理 SCSI 命令的队列、错误处理、设备发现和资源分配。

#### 关键接口与功能：
- **命令提交接口**：
  - 高层驱动通过 `scsi_execute()` 或 `scsi_queue_rq()` 提交命令到中间层。
  ```c
  // 提交 SCSI 命令（drivers/scsi/scsi_lib.c）
  int scsi_execute(struct scsi_device *sdev, const unsigned char *cmd,
                   int data_direction, void *buffer, unsigned bufflen,
                   int timeout, int retries, int flags, int *resid);
  ```

- **队列管理**：
  - 使用 `struct blk_mq_tag_set` 管理多队列请求（现代内核）。
  - 实现 I/O 调度（合并、排序）。

- **错误处理**：
  - 中间层检测超时、设备无响应等问题，触发错误恢复（如 `scsi_eh` 线程）。
  ```c
  // 错误恢复线程（drivers/scsi/scsi_error.c）
  void scsi_error_handler(void) {
      while (!kthread_should_stop()) {
          // 检查错误并尝试恢复（重置设备、重试命令等）
      }
  }
  ```

- **设备发现与热插拔**：
  - 通过 `scsi_scan_host()` 扫描 SCSI 总线，识别新设备。
  - 创建设备节点（如 `/dev/sda`）。

---

### 3. **底层驱动（SCSI Lower Layer / UFS 驱动）**
**代表模块**：UFS 主机控制器驱动（如 `ufs-hcd.c`）、SAS 驱动（如 `mpt3sas`）  
**作用**：将 SCSI 命令转换为硬件特定的操作（如 UFS UPIU、SATA FIS）。

#### 关键接口与实现：
- **SCSI 主机适配器接口**：
  - 每个底层驱动注册一个 `struct Scsi_Host`，表示一个 SCSI 控制器。
  ```c
  // 注册 SCSI 主机（drivers/scsi/hosts.c）
  struct Scsi_Host *scsi_host_alloc(struct scsi_host_template *sht, int privsize);
  ```

- **命令传输接口**：
  - 实现 `struct scsi_host_template` 中的回调函数，处理 SCSI 命令的发送。
  ```c
  // UFS 驱动的传输接口（drivers/ufs/ufshcd.c）
  static struct scsi_host_template ufshcd_driver_template = {
      .module         = THIS_MODULE,
      .name           = "UFS Host Controller",
      .queuecommand   = ufshcd_queuecommand,  // 发送命令到硬件
      .eh_abort_handler = ufshcd_abort,       // 中止命令
  };
  ```

- **UFS 协议处理**：
  - 将 SCSI 命令转换为 **UFS Protocol Information Units (UPIU)**。
  - 处理 UFS 的特定功能（如电源管理、错误恢复）。
  ```c
  // 转换 SCSI 命令到 UPIU（drivers/ufs/ufshcd.c）
  static int ufshcd_queuecommand(struct Scsi_Host *host, struct scsi_cmnd *cmd) {
      // 将 scsi_cmnd 转换为 UPIU 结构体
      struct utp_upiu_req *upiu_req = ...;
      upiu_req->header.dword_0 = cpu_to_be32(UPIU_HEADER_TRANSACTION_TYPE_COMMAND);
      // 填充 LUN、CDB 等字段
  }
  ```

---

### 4. **层间接口总结**
| 层              | 接口/数据结构                  | 功能描述                                                                 |
|------------------|-------------------------------|--------------------------------------------------------------------------|
| **块层 → SCSI 高层** | `struct bio` → `struct scsi_cmnd` | 将块请求（如读/写）转换为 SCSI 命令。                                    |
| **SCSI 高层 → 中间层** | `scsi_execute()`              | 提交 SCSI 命令到中间层队列。                                             |
| **中间层 → 底层驱动** | `struct scsi_host_template`   | 底层驱动实现的回调函数（如 `.queuecommand`），用于实际发送命令到硬件。   |
| **底层驱动 → 硬件**  | UPIU（UFS） / FIS（SATA）      | 将 SCSI 命令编码为硬件协议格式，通过寄存器或 DMA 传输到控制器。          |

---

## 三、关键数据流示例：读取一个磁盘块
1. **用户层**：调用 `read()` 系统调用。
2. **VFS**：将请求传递给文件系统（如 ext4）。
3. **块层**：文件系统生成 `struct bio`，提交到块设备队列。
4. **SCSI 高层驱动（sd.c）**：
   - 将 `bio` 转换为 `struct scsi_cmnd`，填充 `READ_10` 命令。
5. **SCSI 中间层**：
   - 将命令加入队列，处理可能的合并或排序。
6. **UFS 底层驱动**：
   - 调用 `ufshcd_queuecommand()`，将 `scsi_cmnd` 转换为 UPIU。
   - 通过内存映射寄存器或 DMA 将 UPIU 发送到 UFS 控制器。
7. **硬件**：
   - UFS 控制器解析 UPIU，从闪存读取数据，返回响应。

---

## 四、调试与代码定位
### 1. **SCSI 命令跟踪**
- 使用 `scsi_logging` 内核参数启用调试：
  ```bash
  echo 1 > /sys/module/scsi_mod/parameters/logging_level
  ```
- 查看 `dmesg` 输出中的 SCSI 命令和状态。

### 2. **UFS 驱动调试**
- 启用 UFS 调试日志（需内核配置 `CONFIG_SCSI_UFS_DEBUG`）：
  ```c
  // 在代码中手动添加调试打印（drivers/ufs/ufshcd.c）
  ufshcd_dump_regs(hba, 0, UFSHCI_REG_SPACE_SIZE, "UFS Registers:");
  ```

### 3. **关键代码文件**
- **SCSI 高层驱动**：`drivers/scsi/sd.c`
- **SCSI 中间层**：`drivers/scsi/scsi_lib.c`, `drivers/scsi/scsi_error.c`
- **UFS 驱动**：`drivers/ufs/ufshcd.c`, `drivers/ufs/ufs.h`

---

## 五、总结
- **高层驱动**（如 `sd.c`）负责与块层对接，处理通用块设备逻辑。
- **中间层**管理命令队列、错误恢复和设备发现，是 SCSI 子系统的核心逻辑。
- **底层驱动**（如 UFS）实现硬件协议转换，直接操作控制器寄存器或 DMA。
- **接口设计**：通过 `struct scsi_host_template` 和 `struct scsi_cmnd` 实现层间解耦。

这种分层设计使得 Linux 能够支持多种存储协议（SCSI、SATA、UFS 等），同时保持代码的可维护性和扩展性。