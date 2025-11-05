
# 从零写 OS 内核-第三十四篇：PCIe 基础与驱动框架 —— 现代硬件的高速公路

> **“e1000 驱动让我们连上网，但现代网卡早已升级到 PCIe！  
> 今天，我们深入 PCIe 世界，实现配置空间访问、MSI 中断与设备枚举！”**

在上一篇中，我们实现了 **e1000 网卡驱动**，  
但代码仅使用了**传统 PCI 配置机制**（CF8/CFC 端口）。  
而现代硬件（包括 QEMU 模拟的 e1000）实际运行在 **PCI Express（PCIe）** 总线上。

PCIe 不仅是“更快的 PCI”，  
更是**全新的高速串行互连架构**！  

今天，我们就来：  
✅ **解析 PCIe 与传统 PCI 的区别**  
✅ **实现 PCIe 配置空间访问**  
✅ **支持 MSI/MSI-X 中断**  
✅ **构建通用 PCIe 驱动框架**  

让你的 OS 拥有**现代硬件支持能力**！

---

## 🚗 一、PCI vs PCIe：架构革命

### 传统 PCI（并行总线）：
- **共享总线**：所有设备争用同一带宽  
- **33/66 MHz 时钟**：最大 133 MB/s 带宽  
- **中断共享**：IRQ 线需多个设备共享  
- **配置机制**：通过 `0xCF8/0xCFC` 端口访问  

### PCIe（串行点对点）：
- **点对点连接**：每个设备独占通道（Lane）  
- **高带宽**：x1 通道 = 250 MB/s（Gen1），x16 = 4 GB/s  
- **消息中断**：MSI/MSI-X 替代传统 IRQ  
- **配置空间扩展**：从 256B → 4KB  

| 特性 | PCI | PCIe |
|------|-----|------|
| **拓扑** | 共享总线 | 树状拓扑（Root Complex + Switches）|
| **配置空间** | 256 字节 | 4KB（含扩展 Capability）|
| **中断** | 4 条 IRQ 线 | MSI/MSI-X（内存写中断）|
| **热插拔** | 不支持 | 原生支持 |

> 💡 **PCIe 向前兼容 PCI 配置空间前 256 字节**，但扩展功能需新机制！

---

## 🔌 二、PCIe 配置空间访问

### 1. **传统 PCI 配置（端口 I/O）**
```c
// 仅支持前 256 字节，且需串行访问
uint32_t pci_config_read(uint8_t bus, uint8_t slot, uint8_t func, uint8_t offset) {
    uint32_t address = (1 << 31) | (bus << 16) | (slot << 11) | (func << 8) | (offset & 0xFC);
    outl(0xCF8, address);
    return inl(0xCFC + (offset & 0x3));
}
```

### 2. **PCIe 配置（MMIO 方式）**
现代系统通过 **MCFG 表**（ACPI）提供 **ECAM**（Enhanced Configuration Access Mechanism）基地址：

```c
// ACPI MCFG 表结构
struct acpi_mcfg {
    char signature[4];      // "MCFG"
    uint32_t length;
    uint8_t revision;
    uint8_t checksum;
    char oem_id[6];
    char oem_table_id[8];
    uint32_t oem_revision;
    uint32_t creator_id;
    uint32_t creator_revision;
    uint64_t reserved;
    struct mcfg_allocation {
        uint64_t base_address;
        uint16_t segment_group;
        uint8_t start_bus;
        uint8_t end_bus;
        uint32_t reserved;
    } allocations[0];
};
```

### 3. **ECAM 地址计算**
```c
// ECAM 偏移 = (Bus << 20) | (Device << 15) | (Function << 12) | Offset
uint32_t ecam_read(uint64_t ecam_base, 
                   uint8_t bus, uint8_t dev, uint8_t func, uint16_t offset) {
    uint64_t addr = ecam_base + 
                   ((uint64_t)bus << 20) + 
                   ((uint64_t)dev << 15) + 
                   ((uint64_t)func << 12) + 
                   offset;
    return *(volatile uint32_t*)(addr + KERNEL_VIRTUAL_BASE);
}
```

> 🔑 **ECAM 允许并行访问，且支持完整 4KB 配置空间**！

---

## 📦 三、PCIe Capability 链

PCIe 设备通过 **Capability 链** 暴露高级功能：

### 1. **Capability ID 列表**
| ID | 功能 |
|----|------|
| `0x01` | PM（电源管理）|
| `0x05` | MSI（消息信号中断）|
| `0x10` | PCIe（设备能力）|
| `0x11` | MSI-X |
| `0x12` | Vendor-Specific |

### 2. **遍历 Capability 链**
```c
void pci_scan_capabilities(uint8_t bus, uint8_t dev, uint8_t func) {
    uint16_t status = pci_read_word(bus, dev, func, 0x06);
    if (!(status & 0x10)) return; // 无 Capability 链
    
    uint8_t cap_ptr = pci_read_byte(bus, dev, func, 0x34);
    while (cap_ptr) {
        uint8_t cap_id = pci_read_byte(bus, dev, func, cap_ptr);
        uint8_t next_ptr = pci_read_byte(bus, dev, func, cap_ptr + 1);
        
        switch (cap_id) {
            case 0x05: // MSI
                handle_msi_capability(bus, dev, func, cap_ptr);
                break;
            case 0x11: // MSI-X
                handle_msix_capability(bus, dev, func, cap_ptr);
                break;
            case 0x10: // PCIe
                handle_pcie_capability(bus, dev, func, cap_ptr);
                break;
        }
        cap_ptr = next_ptr;
    }
}
```

---

## ⚡ 四、MSI 中断：告别 IRQ 共享

### MSI 优势：
- **无共享冲突**：每个设备独立中断向量  
- **低延迟**：直接内存写，无需 IRQ 线  
- **多向量支持**：一个设备可申请多个中断  

### 1. **MSI Capability 结构**
```c
// MSI Capability 寄存器（偏移 cap_ptr）
#define MSI_CAP_MSG_CONTROL  (cap_ptr + 2)
#define MSI_CAP_MSG_ADDR_LO  (cap_ptr + 4)
#define MSI_CAP_MSG_ADDR_HI  (cap_ptr + 8)
#define MSI_CAP_MSG_DATA     (cap_ptr + 12)
```

### 2. **启用 MSI**
```c
void enable_msi(uint8_t bus, uint8_t dev, uint8_t func, uint8_t cap_ptr) {
    // 1. 读取 MSI 控制寄存器
    uint16_t msi_ctrl = pci_read_word(bus, dev, func, MSI_CAP_MSG_CONTROL);
    
    // 2. 设置中断向量（假设 IRQ 16-23 可用）
    uint32_t msg_addr = 0xFEE00000 | (0x00 << 12); // LAPIC 地址 + 目标核
    uint16_t msg_data = 0x00 | (16 << 0);           // 向量号 16
    
    // 3. 写入 MSI 寄存器
    pci_write_dword(bus, dev, func, MSI_CAP_MSG_ADDR_LO, msg_addr);
    if (msi_ctrl & 0x80) { // 64-bit 地址
        pci_write_dword(bus, dev, func, MSI_CAP_MSG_ADDR_HI, 0x00000000);
        pci_write_word(bus, dev, func, MSI_CAP_MSG_DATA, msg_data);
    } else {
        pci_write_word(bus, dev, func, MSI_CAP_MSG_DATA, msg_data);
    }
    
    // 4. 启用 MSI
    msi_ctrl |= 0x01;
    pci_write_word(bus, dev, func, MSI_CAP_MSG_CONTROL, msi_ctrl);
    
    // 5. 屏蔽传统 INTx
    uint16_t devctl = pci_read_word(bus, dev, func, cap_ptr + 0x08);
    devctl |= 0x400; // INTx Disable
    pci_write_word(bus, dev, func, cap_ptr + 0x08, devctl);
}
```

### 3. **注册 MSI 中断处理程序**
```c
// 将向量 16 映射到 e1000 中断处理程序
idt_set_gate(32 + 16, (uint32_t)e1000_msi_handler, 0x08, 0x8E);
```

> ✅ **MSI 让每个设备拥有专属中断，彻底解决 IRQ 共享问题**！

---

## 🏗️ 五、通用 PCIe 驱动框架

### 1. **设备结构抽象**
```c
struct pcie_device {
    uint8_t bus, dev, func;
    uint16_t vendor_id, device_id;
    uint8_t class_code[3];
    uint64_t bar[6];        // Base Address Registers
    uint8_t msi_enabled;
    uint8_t msix_enabled;
    void *driver_data;
};

// 全局设备列表
struct pcie_device pcie_devices[MAX_DEVICES];
int pcie_device_count = 0;
```

### 2. **设备枚举**
```c
void pcie_enumerate() {
    for (int bus = 0; bus < 256; bus++) {
        for (int dev = 0; dev < 32; dev++) {
            // 检查是否为多功能设备
            uint16_t vendor = pci_read_word(bus, dev, 0, 0x00);
            if (vendor == 0xFFFF) continue;
            
            int max_func = ((pci_read_byte(bus, dev, 0, 0x0E) & 0x80) ? 8 : 1);
            for (int func = 0; func < max_func; func++) {
                vendor = pci_read_word(bus, dev, func, 0x00);
                if (vendor == 0xFFFF) continue;
                
                uint16_t device = pci_read_word(bus, dev, func, 0x02);
                uint32_t class_rev = pci_read_dword(bus, dev, func, 0x08);
                
                // 保存设备信息
                struct pcie_device *pdev = &pcie_devices[pcie_device_count++];
                pdev->bus = bus; pdev->dev = dev; pdev->func = func;
                pdev->vendor_id = vendor; pdev->device_id = device;
                pdev->class_code[0] = (class_rev >> 24) & 0xFF;
                pdev->class_code[1] = (class_rev >> 16) & 0xFF;
                pdev->class_code[2] = (class_rev >> 8) & 0xFF;
                
                // 读取 BAR
                for (int i = 0; i < 6; i++) {
                    uint32_t bar = pci_read_dword(bus, dev, func, 0x10 + i*4);
                    if (bar == 0) continue;
                    
                    // 判断内存/IO 空间
                    if (bar & 0x01) {
                        // IO 空间（传统 PCI）
                        pdev->bar[i] = bar & 0xFFFFFFFC;
                    } else {
                        // 内存空间
                        uint8_t type = (bar >> 1) & 0x03;
                        if (type == 0x02) {
                            // 64-bit BAR
                            uint32_t bar_high = pci_read_dword(bus, dev, func, 0x14 + i*4);
                            pdev->bar[i] = ((uint64_t)bar_high << 32) | (bar & 0xFFFFFFF0);
                            i++; // 跳过下一个 BAR
                        } else {
                            pdev->bar[i] = bar & 0xFFFFFFF0;
                        }
                    }
                }
                
                // 扫描 Capability
                pci_scan_capabilities(bus, dev, func);
                
                // 尝试匹配驱动
                match_driver(pdev);
            }
        }
    }
}
```

### 3. **驱动匹配机制**
```c
struct pcie_driver {
    char name[32];
    uint16_t vendor_id;
    uint16_t device_id;
    int (*probe)(struct pcie_device *dev);
};

// 注册驱动
static struct pcie_driver e1000_driver = {
    .name = "e1000",
    .vendor_id = 0x8086,
    .device_id = 0x100E,
    .probe = e1000_probe,
};

void match_driver(struct pcie_device *dev) {
    if (dev->vendor_id == e1000_driver.vendor_id &&
        dev->device_id == e1000_driver.device_id) {
        e1000_driver.probe(dev);
    }
}
```

---

## 🧪 六、测试：QEMU PCIe 环境

### 1. **QEMU 启动命令**
```bash
# QEMU 默认使用 PCIe 总线（即使指定 -device e1000）
qemu-system-i386 -kernel kernel.bin -hda disk.img \
                 -netdev user,id=n1 -device e1000,netdev=n1
```

### 2. **内核启动日志**
```
[PCIe] Found 82540EM Gigabit Ethernet Controller (8086:100E)
[PCIe] BAR0: MMIO at 0xFEBC0000 (128KB)
[PCIe] MSI enabled (vector 16)
[e1000] Driver loaded successfully
```

### 3. **验证 MSI 中断**
- 传统 IRQ 10 不再触发  
- 向量 48（32+16）的中断处理程序被调用  

✅ **PCIe + MSI 完整工作链**！

---

## ⚠️ 七、高级话题

1. **MSI-X 支持**  
   - 支持更多中断向量（>32）  
   - 表结构存储在设备内存中  

2. **PCIe 链路训练**  
   - 自动协商通道宽度（x1/x4/x8）  
   - 速率协商（Gen1/Gen2/Gen3）  

3. **ATS（Address Translation Service）**  
   - 与 IOMMU 协同，支持虚拟化  

4. **SR-IOV（Single Root I/O Virtualization）**  
   - 物理设备虚拟化为多个 VF（Virtual Function）  

> 💡 **现代数据中心依赖 PCIe 高级特性实现高性能 I/O 虚拟化**！

---

## 💬 写在最后

PCIe 是现代计算的**硬件骨干**。  
它不仅是更快的总线，  
更是**高性能、低延迟、可扩展 I/O 的基石**。

今天你启用的第一个 MSI 中断，  
正是无数 NVMe SSD、GPU、100G 网卡高效工作的起点。

> 🌟 **理解 PCIe，就是理解现代硬件的脉搏**。

---

📬 **动手挑战**：  
为 e1000 驱动添加 MSI-X 支持，并测试多队列中断。  
欢迎在评论区分享你的 PCIe 设备枚举日志！

👇 下一篇你想看：**NVMe SSD 驱动**，还是 **IOMMU 与 DMA 保护**？

---

**#操作系统 #内核开发 #PCIe #MSI #设备驱动 #硬件抽象 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“pcie”**，获取：
> - 完整 PCIe 枚举与 MSI 启用代码
> - ACPI MCFG 解析模板
> - 通用 PCIe 驱动框架
