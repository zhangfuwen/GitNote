# 从零写 OS 内核-第二十九篇：鼠标驱动 —— 为你的图形界面添加指针控制

> **“键盘虽高效，但图形界面离不开鼠标。  
> 今天，我们实现 PS/2 鼠标驱动，让你的窗口管理器支持点击与拖拽！”**

在上一篇中，我们实现了**键盘驱动的窗口管理器**，  
但真正的图形界面需要**鼠标支持**：  
- 点击按钮  
- 拖拽窗口  
- 精确定位  

而 x86 PC 最经典的鼠标接口是 **PS/2**（通过 i8042 键盘控制器复用）。  

今天，我们就来：  
✅ **解析 PS/2 鼠标协议**  
✅ **实现 i8042 控制器驱动**  
✅ **注册鼠标设备到 devfs**  
✅ **在 Display Server 中集成鼠标事件**  

让你的 OS 拥有**完整的指针控制能力**！

---

## 🔌 一、PS/2 鼠标硬件基础

### PS/2 接口特点：
- **6 针 Mini-DIN 接口**（鼠标通常为绿色）  
- **串行通信**：时钟线（CLK） + 数据线（DATA）  
- **由 i8042 键盘控制器管理**（端口 0x60/0x64）  

### i8042 端口：
| 端口 | 名称 | 作用 |
|------|------|------|
| `0x60` | 数据端口 | 读写鼠标/键盘数据 |
| `0x64` | 状态端口 | 读取控制器状态 |
| `0x64` | 命令端口 | 发送控制器命令 |

### 状态寄存器位：
- **bit 0 (OBF)**：1 = 输出缓冲区满（可读数据）  
- **bit 1 (IBF)**：1 = 输入缓冲区满（勿写命令）  
- **bit 5 (AUX_BSY)**：1 = 鼠标忙  

> 💡 **PS/2 鼠标使用 AUX 端口（与键盘主端口分离）**！

---

## 📡 二、PS/2 鼠标协议

### 初始化流程：
1. **检测鼠标存在**  
2. **启用鼠标**  
3. **设置采样率/分辨率**  
4. **启用数据报告**  

### 鼠标数据包格式（3 字节）：
```
Byte 0:  [Y溢出][X溢出][Y符号][X符号][1][M][R][L]
Byte 1:  X 轴相对位移（带符号）
Byte 2:  Y 轴相对位移（带符号）
```
- **L/R/M**：左/右/中键状态（1=按下）  
- **X/Y 符号**：1=负方向（左/上）  
- **溢出位**：位移超出 -255~255 范围  

> ✅ **我们只处理标准 3 字节包**（忽略滚轮等扩展）。

---

## ⚙️ 三、i8042 控制器驱动

### 1. **等待控制器就绪**
```c
// drivers/ps2.c
void ps2_wait_write() {
    while (inb(0x64) & 0x02); // 等待 IBF=0
}

void ps2_wait_read() {
    int timeout = 100000;
    while ((inb(0x64) & 0x01) == 0 && timeout--); // 等待 OBF=1
}
```

### 2. **发送 AUX 命令**
```c
uint8_t ps2_send_aux_command(uint8_t cmd) {
    ps2_wait_write();
    outb(0x64, 0xD4); // 发送 AUX 命令前缀
    
    ps2_wait_write();
    outb(0x60, cmd);
    
    ps2_wait_read();
    return inb(0x60); // 返回设备响应
}
```

### 3. **初始化鼠标**
```c
bool ps2_mouse_init() {
    // 1. 检测 AUX 端口存在
    ps2_wait_write();
    outb(0x64, 0xA8); // 激活 AUX 端口
    
    // 2. 检测鼠标存在
    uint8_t ack = ps2_send_aux_command(0xF2); // 获取设备 ID
    if (ack != 0xFA) return false; // 不是 ACK
    
    ps2_wait_read();
    uint8_t id = inb(0x60);
    if (id == 0x00 || id == 0xFF) return false; // 无鼠标或错误
    
    // 3. 设置默认参数
    ps2_send_aux_command(0xF6); // 默认设置
    ps2_send_aux_command(0xF4); // 启用数据报告
    
    return true;
}
```

---

## 🖱️ 四、鼠标中断处理

### 1. **注册 IRQ12 中断**
```c
void ps2_init() {
    if (!ps2_mouse_init()) {
        printk("PS/2 Mouse not detected\n");
        return;
    }
    
    // IRQ12 = 鼠标中断
    idt_set_gate(44, (uint32_t)mouse_irq_handler, 0x08, 0x8E);
    pic_enable_irq(12);
}
```

### 2. **中断处理程序**
```c
void mouse_irq_handler() {
    // 1. 读取数据
    uint8_t status = inb(0x64);
    if (!(status & 0x20)) return; // 不是鼠标中断
    
    uint8_t data = inb(0x60);
    
    // 2. 加入环形缓冲区
    mouse_buffer[mouse_head] = data;
    mouse_head = (mouse_head + 1) % MOUSE_BUFFER_SIZE;
    
    // 3. 唤醒鼠标线程
    if (mouse_thread) {
        wake_up_process(mouse_thread);
    }
    
    // 4. 发送 EOI
    outb(0xA0, 0x20); // 从 PIC
    outb(0x20, 0x20); // 主 PIC
}
```

> 🔑 **中断只收集数据，解析在用户态线程完成**（避免内核复杂逻辑）！

---

## 📁 五、鼠标设备文件（/dev/input/mouse0）

### 1. **注册字符设备**
```c
static struct file_operations mouse_fops = {
    .read = mouse_read,
    .poll = mouse_poll,
};

void mouse_register_device() {
    struct device *dev = kmalloc(sizeof(struct device));
    strcpy(dev->name, "mouse0");
    dev->devt = mkdev(MAJOR_INPUT, 0);
    device_register(dev);
    register_chrdev(MAJOR_INPUT, "input", &mouse_fops);
}
```

### 2. **read 系统调用**
```c
ssize_t mouse_read(struct file *file, char *buf, size_t count) {
    if (count < sizeof(struct mouse_event)) {
        return -1;
    }
    
    // 等待事件
    while (mouse_tail == mouse_head) {
        sleep_on(&mouse_wait_queue);
    }
    
    // 解析 3 字节包
    if ((mouse_head - mouse_tail) % MOUSE_BUFFER_SIZE >= 3) {
        uint8_t byte0 = mouse_buffer[mouse_tail];
        uint8_t byte1 = mouse_buffer[(mouse_tail + 1) % MOUSE_BUFFER_SIZE];
        uint8_t byte2 = mouse_buffer[(mouse_tail + 2) % MOUSE_BUFFER_SIZE];
        
        struct mouse_event event = {
            .buttons = (byte0 & 0x07),
            .dx = (int8_t)byte1,
            .dy = -(int8_t)byte2, // Y 轴反转（屏幕坐标）
        };
        
        // 复制到用户空间
        copy_to_user(buf, &event, sizeof(event));
        
        mouse_tail = (mouse_tail + 3) % MOUSE_BUFFER_SIZE;
        return sizeof(event);
    }
    
    return -1;
}
```

### 3. **鼠标事件结构**
```c
// include/mouse.h
struct mouse_event {
    uint8_t buttons; // 位 0=左键, 1=右键, 2=中键
    int8_t dx;       // X 轴相对位移
    int8_t dy;       // Y 轴相对位移
};
```

---

## 🖥️ 六、Display Server 集成鼠标

### 1. **启动鼠标监听线程**
```c
// user/display_server.c
void start_mouse_thread() {
    if (fork() == 0) {
        int mouse_fd = open("/dev/input/mouse0", O_RDONLY);
        struct mouse_event event;
        
        while (1) {
            if (read(mouse_fd, &event, sizeof(event)) > 0) {
                // 发送事件到 Window Manager
                send_mouse_event_to_wm(&event);
            }
        }
    }
}
```

### 2. **维护绝对坐标**
```c
// Window Manager 中
static int mouse_x = 0, mouse_y = 0;
static int screen_width = 1024, screen_height = 768;

void handle_mouse_event(struct mouse_event *event) {
    // 更新绝对坐标
    mouse_x = min(max(0, mouse_x + event->dx), screen_width - 1);
    mouse_y = min(max(0, mouse_y + event->dy), screen_height - 1);
    
    // 检查按钮事件
    if (event->buttons & 1) { // 左键按下
        struct window *win = find_window_at(mouse_x, mouse_y);
        if (win && !wm_state.dragging) {
            // 开始拖拽
            wm_state.dragging = true;
            wm_state.drag_start_x = mouse_x - win->x;
            wm_state.drag_start_y = mouse_y - win->y;
            set_focus(win);
        }
    } else {
        // 左键释放
        wm_state.dragging = false;
    }
    
    // 处理拖拽
    if (wm_state.dragging) {
        struct window *win = wm_state.focused_window;
        if (win) {
            move_window(win->id, 
                       mouse_x - wm_state.drag_start_x,
                       mouse_y - wm_state.drag_start_y);
        }
    }
    
    // 重绘鼠标指针（简化：用十字）
    draw_mouse_cursor(mouse_x, mouse_y);
}
```

---

## 🧪 七、测试：鼠标控制窗口

### 操作流程：
1. **系统启动** → 检测到 PS/2 鼠标  
2. **移动鼠标** → 屏幕显示十字指针  
3. **点击窗口标题栏** → 窗口获得焦点（标题变蓝）  
4. **拖拽标题栏** → 窗口跟随鼠标移动  
5. **点击关闭提示** → 窗口关闭  

### QEMU 测试命令：
```bash
qemu-system-i386 -kernel kernel.bin -hda disk.img -serial stdio -device usb-mouse
# 或使用 PS/2 鼠标
qemu-system-i386 -kernel kernel.bin -hda disk.img -serial stdio -usb -device usb-kbd -device usb-mouse
```

> 💡 **QEMU 默认模拟 USB 鼠标，但 i8042 驱动可兼容**。

---

## ⚠️ 八、高级话题

1. **滚轮支持**  
   - 扩展为 4 字节包  
   - `byte3` 包含滚轮位移

2. **多鼠标支持**  
   - `/dev/input/mouse0`, `/dev/input/mouse1`  
   - 通过设备树区分

3. **鼠标加速**  
   - 根据移动速度动态调整 dx/dy

4. **USB 鼠标驱动**  
   - 实现 USB HID 类驱动  
   - 支持现代 USB 鼠标

> 💡 **Linux 的 `psmouse` 驱动支持从 PS/2 到 USB 的无缝切换**！

---

## 💬 写在最后

鼠标驱动是图形界面的**最后一块拼图**。  
它将物理移动转化为屏幕坐标，  
让用户能直观地与窗口交互。

今天你实现的第一个鼠标事件，  
正是无数 GUI 交互的起点。

> 🌟 **指针的每一次移动，都是人机对话的优雅舞蹈。**

---

📬 **动手挑战**：  
添加鼠标右键菜单（在窗口标题栏右键弹出菜单）。  
欢迎在评论区分享你的鼠标驱动调试技巧！

👇 下一篇你想看：**USB 鼠标驱动**，还是 **触摸屏支持**？

---

**#操作系统 #内核开发 #鼠标驱动 #PS2 #i8042 #图形界面 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“mouse”**，获取：
> - 完整 PS/2 鼠标驱动代码
> - 鼠标事件解析与坐标转换模板
> - QEMU 鼠标测试配置指南
