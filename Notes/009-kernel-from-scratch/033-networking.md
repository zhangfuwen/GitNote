# 从零写 OS 内核-第三十三篇：e1000 网卡驱动 —— 让你的 OS 接入网络世界！

> **“没有网络的操作系统，只是信息孤岛。  
> 今天，我们实现 Intel e1000 网卡驱动，让 MyOS 能 ping 通外部世界！”**

在前面的篇章中，我们构建了完整的桌面系统：  
✅ **窗口管理器**  
✅ **LVGL GUI**  
✅ **文件系统**  

但所有操作都局限在本地——**无法访问互联网**，**无法与其他机器通信**。  

真正的操作系统必须支持**网络功能**！  
而 **Intel 8254x 系列（e1000）** 是 QEMU 完美模拟的经典网卡，  
也是学习网络驱动的最佳起点。

今天，我们就来：  
✅ **解析 e1000 硬件架构**  
✅ **实现 MMIO 寄存器访问**  
✅ **配置接收/发送描述符**  
✅ **处理网卡中断**  
✅ **发送第一个 ICMP Ping 包**  

让你的 OS 拥有**真正的网络能力**！

---

## 🌐 一、e1000 硬件基础

### 为什么选择 e1000？
- **QEMU 原生支持**：`-netdev user,id=n1 -device e1000,netdev=n1`  
- **文档齐全**：Intel 官方《8254x Software Developer’s Manual》  
- **架构经典**：DMA 描述符 + 中断驱动  

### e1000 关键组件：
| 组件 | 作用 |
|------|------|
| **MMIO 寄存器** | 控制网卡（基地址由 PCI 配置空间提供）|
| **接收描述符环** | 网卡存放接收到的帧 |
| **发送描述符环** | CPU 提交待发送的帧 |
| **中断系统** | 通知 CPU 帧接收/发送完成 |

> 💡 **e1000 通过 DMA 直接读写物理内存，无需 CPU 拷贝数据**！

---

## 🔌 二、PCI 设备探测

### 1. **扫描 PCI 总线**
e1000 的 PCI Vendor ID = `0x8086`（Intel），Device ID = `0x100E`（QEMU 模拟）

```c
// drivers/pci.c
bool pci_scan_for_e1000() {
    for (int bus = 0; bus < 256; bus++) {
        for (int slot = 0; slot < 32; slot++) {
            for (int func = 0; func < 8; func++) {
                uint16_t vendor = pci_read_word(bus, slot, func, 0x00);
                if (vendor == 0xFFFF) continue; // 无效设备
                
                uint16_t device = pci_read_word(bus, slot, func, 0x02);
                if (vendor == 0x8086 && device == 0x100E) {
                    // 找到 e1000
                    e1000_bus = bus;
                    e1000_slot = slot;
                    e1000_func = func;
                    return true;
                }
            }
        }
    }
    return false;
}
```

### 2. **获取 BAR（Base Address Register）**
```c
void e1000_init_mmio() {
    // 读取 BAR0（MMIO 基地址）
    uint32_t bar0 = pci_read_dword(e1000_bus, e1000_slot, e1000_func, 0x10);
    uint32_t mmio_base = bar0 & 0xFFFFFFF0; // 清除标志位
    
    // 映射到内核虚拟地址
    e1000_mmio_vaddr = (void*)(mmio_base + KERNEL_VIRTUAL_BASE);
    map_range(mmio_base, (uint32_t)e1000_mmio_vaddr, 128 * 1024, 
              PAGE_PRESENT | PAGE_RW | PAGE_GLOBAL);
}
```

> 🔑 **MMIO 寄存器通过内存映射访问，无需 inb/outb**！

---

## 📡 三、e1000 寄存器操作

### 1. **寄存器定义**
```c
// drivers/e1000.h
#define E1000_REG_CTRL     0x0000  // 控制寄存器
#define E1000_REG_STATUS   0x0008  // 状态寄存器
#define E1000_REG_ICR      0x00C0  // 中断原因
#define E1000_REG_IMS      0x00D0  // 中断屏蔽
#define E1000_REG_RCTL     0x0100  // 接收控制
#define E1000_REG_TCTL     0x0400  // 发送控制
#define E1000_REG_RDBAL    0x2800  // 接收描述符基地址低 32 位
#define E1000_REG_TDBAL    0x3800  // 发送描述符基地址低 32 位
```

### 2. **读写宏**
```c
static inline uint32_t e1000_read(uint32_t reg) {
    return *(volatile uint32_t*)((char*)e1000_mmio_vaddr + reg);
}

static inline void e1000_write(uint32_t reg, uint32_t value) {
    *(volatile uint32_t*)((char*)e1000_mmio_vaddr + reg) = value;
}
```

---

## 📦 四、接收/发送描述符

### 1. **描述符结构**
```c
// 接收描述符（64 字节）
struct e1000_rx_desc {
    uint64_t buffer_addr;    // 物理地址（2KB 缓冲区）
    uint16_t length;         // 接收长度
    uint16_t checksum;
    uint8_t status;          // DD=1 表示有效
    uint8_t errors;
    uint16_t special;
};

// 发送描述符（16 字节）
struct e1000_tx_desc {
    uint64_t buffer_addr;    // 物理地址
    uint16_t length;
    uint8_t cso;             // Checksum offset
    uint8_t cmd;             // RS=1 报告状态, EOP=1 结束包
    uint8_t status;          // DD=1 表示完成
    uint8_t css;
    uint16_t special;
};
```

### 2. **初始化描述符环**
```c
void e1000_init_rx_ring() {
    // 1. 分配 32 个接收缓冲区（每个 2KB）
    for (int i = 0; i < RX_DESC_COUNT; i++) {
        rx_buffers[i] = buddy_alloc_pages(1); // 2KB = 半页
        rx_descs[i].buffer_addr = (uint64_t)rx_buffers[i];
        rx_descs[i].status = 0;
    }
    
    // 2. 设置接收描述符基地址
    e1000_write(E1000_REG_RDBAL, (uint32_t)rx_descs);
    e1000_write(E1000_REG_RDBAH, 0);
    e1000_write(E1000_REG_RDLEN, RX_DESC_COUNT * sizeof(struct e1000_rx_desc));
    e1000_write(E1000_REG_RDH, 0);
    e1000_write(E1000_REG_RDT, RX_DESC_COUNT - 1); // 尾指针
    
    // 3. 启用接收
    e1000_write(E1000_REG_RCTL, 
                E1000_RCTL_EN |     // 启用接收
                E1000_RCTL_BAM |    // 广播接收
                E1000_RCTL_LBM_NONE | 
                E1000_RCTL_RDMTS_HALF |
                E1000_RCTL_BSIZE_2048);
}
```

---

## ⚡ 五、中断处理

### 1. **启用中断**
```c
void e1000_enable_interrupts() {
    // 屏蔽所有中断
    e1000_write(E1000_REG_IMC, 0xFFFFFFFF);
    
    // 仅启用接收/发送完成中断
    e1000_write(E1000_REG_IMS, 
                E1000_IMS_RXT0 |    // 接收超时
                E1000_IMS_RXDMT0 |  // 接收描述符最小阈值
                E1000_IMS_TXDW);    // 发送描述符写回
    
    // 允许 PCI 中断
    pci_write_word(e1000_bus, e1000_slot, e1000_func, 0x3C, 0x000A); // IRQ 10
    pic_enable_irq(10);
}
```

### 2. **中断处理程序**
```c
void e1000_irq_handler() {
    uint32_t icr = e1000_read(E1000_REG_ICR); // 读取中断原因（自动清零）
    
    if (icr & (E1000_ICR_RXT0 | E1000_ICR_RXDMT0)) {
        // 处理接收帧
        e1000_handle_rx();
    }
    
    if (icr & E1000_ICR_TXDW) {
        // 处理发送完成
        e1000_handle_tx();
    }
    
    // 发送 EOI
    outb(0xA0, 0x20);
    outb(0x20, 0x20);
}
```

---

## 📤 六、发送第一个 Ping 包

### 1. **构建 ICMP Echo Request**
```c
void send_ping(uint32_t dst_ip) {
    // 1. 分配发送缓冲区
    uint8_t *packet = buddy_alloc_pages(1); // 2KB 足够
    
    // 2. 构建以太网帧头
    struct eth_header *eth = (void*)packet;
    memcpy(eth->dst, "\xFF\xFF\xFF\xFF\xFF\xFF", 6); // 广播
    memcpy(eth->src, e1000_mac, 6);
    eth->type = 0x0800; // IPv4
    
    // 3. 构建 IP 头
    struct ip_header *ip = (void*)(packet + 14);
    ip->version_ihl = 0x45;
    ip->tos = 0;
    ip->total_length = htons(28 + 20); // ICMP + IP
    ip->id = htons(1);
    ip->frag_off = 0;
    ip->ttl = 64;
    ip->protocol = 1; // ICMP
    ip->src_ip = htonl(0x0A000001); // 10.0.0.1
    ip->dst_ip = htonl(dst_ip);
    ip->checksum = 0;
    ip->checksum = ip_checksum(ip, 20);
    
    // 4. 构建 ICMP 头
    struct icmp_header *icmp = (void*)(packet + 14 + 20);
    icmp->type = 8; // Echo Request
    icmp->code = 0;
    icmp->checksum = 0;
    icmp->id = htons(0x1234);
    icmp->seq = htons(1);
    memcpy(icmp->data, "MyOS Ping", 10);
    icmp->checksum = icmp_checksum(icmp, 18);
    
    // 5. 提交到发送描述符
    tx_descs[tx_tail].buffer_addr = (uint64_t)packet;
    tx_descs[tx_tail].length = 14 + 20 + 18;
    tx_descs[tx_tail].cmd = E1000_TXD_CMD_RS | E1000_TXD_CMD_EOP;
    tx_descs[tx_tail].status = 0;
    
    // 6. 更新尾指针
    tx_tail = (tx_tail + 1) % TX_DESC_COUNT;
    e1000_write(E1000_REG_TDT, tx_tail);
}
```

### 2. **QEMU 测试命令**
```bash
# 启用 e1000 网卡 + 用户模式网络（自动分配 10.0.2.15）
qemu-system-i386 -kernel kernel.bin -hda disk.img \
                 -netdev user,id=n1 -device e1000,netdev=n1

# 在 MyOS 中 ping 网关（QEMU 默认网关 10.0.2.2）
ping 10.0.2.2
```

---

## 🧪 七、测试：接收 Ping 回复

### 1. **处理接收帧**
```c
void e1000_handle_rx() {
    while (rx_descs[rx_head].status & 0x01) { // DD=1
        uint8_t *buffer = rx_buffers[rx_head];
        uint16_t length = rx_descs[rx_head].length;
        
        // 检查是否为 ICMP Echo Reply
        if (is_icmp_echo_reply(buffer, length)) {
            printk("Ping reply received!\n");
        }
        
        // 重新挂载缓冲区
        rx_descs[rx_head].status = 0;
        rx_head = (rx_head + 1) % RX_DESC_COUNT;
        e1000_write(E1000_REG_RDT, rx_head - 1);
    }
}
```

### 2. **运行效果**
```
MyOS# ping 10.0.2.2
Sending ping to 10.0.2.2...
Ping reply received!
```

✅ **e1000 驱动成功收发网络包**！

---

## ⚠️ 八、关键注意事项

1. **物理地址 vs 虚拟地址**  
   - 描述符中的 `buffer_addr` 必须是**物理地址**  
   - 通过 `virt_to_phys()` 转换

2. **缓存一致性**  
   - x86 通常无需显式 flush cache（WC 内存类型）  
   - 但需确保描述符更新对设备可见

3. **描述符环边界**  
   - 头/尾指针需取模运算  
   - 避免写满整个环（留一个空描述符）

4. **错误处理**  
   - 检查 `status` 中的错误位（如 CRC 错误）  
   - 重置网卡（极端情况）

> 💡 **Linux 的 e1000 驱动（drivers/net/ethernet/intel/e1000）是绝佳参考**！

---

## 💬 写在最后

e1000 驱动是网络栈的**第一块基石**。  
它将你的 OS 从孤岛连接到广阔网络，  
为后续的 **TCP/IP、HTTP、SSH** 铺平道路。

今天你发送的第一个 Ping 包，  
正是无数网络通信的起点。

> 🌟 **网络的意义，不在于技术，而在于连接**。

---

📬 **动手挑战**：  
实现 ARP 请求/应答，让 MyOS 能通过 MAC 地址通信。  
欢迎在评论区分享你的网络抓包截图！

👇 下一篇你想看：**ARP 协议实现**，还是 **UDP 与 DHCP 客户端**？

---

**#操作系统 #内核开发 #e1000 #网卡驱动 #网络 #PCI #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“e1000”**，获取：
> - 完整 e1000 驱动代码（含 PCI 扫描、描述符管理）
> - ICMP Ping 实现模板
> - QEMU 网络测试配置指南
