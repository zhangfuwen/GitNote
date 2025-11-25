# 从零写 OS 内核-第二十四篇：Display Server 与 Client —— 构建共享内存图形架构

> **“Framebuffer 让你能画像素，但多个应用如何共享屏幕？  
> 今天，我们实现 Display Server + Client 架构，通过共享内存实现高效图形合成！”**

在上一篇中，我们实现了 Framebuffer 驱动，用户程序可直接操作像素。  
但这带来严重问题：  
- **多个程序同时写屏幕 → 画面撕裂、内容混杂**  
- **无窗口概念 → 无法实现 GUI**  
- **无权限控制 → 任意程序可覆盖整个屏幕**

真正的图形系统需要 **Display Server（显示服务）**：  
- **唯一拥有 Framebuffer 写权限**  
- **接收 Client（客户端）的绘图请求**  
- **合成多个窗口并上屏**  

而 Client 通过 **共享内存** 高效提交绘图内容。  

今天，我们就来：  
✅ **设计 Display Server 架构**  
✅ **实现共享内存机制（shm）**  
✅ **构建 Client 绘图 API**  
✅ **演示多窗口合成上屏**  

让你的 OS 拥有**现代图形系统雏形**！

---

## 🖥️ 一、Display Server 架构设计

### 核心组件：
| 组件 | 职责 |
|------|------|
| **Display Server** | 管理屏幕、合成窗口、处理输入 |
| **Client** | 应用程序（如 Terminal、Painter）|
| **Shared Memory** | Client 绘制内容 → Server 读取合成 |
| **IPC 通道** | Client 发送命令（创建窗口、提交帧）|

### 数据流：
```
+------------+       +----------------+       +--------------+
|  Client 1  |       |  Client 2      |       |  ...         |
| (Terminal) |       | (Painter)      |       |              |
+-----+------+       +-------+--------+       +--------------+
      |                      |
      | 1. 绘制到 shm buffer | 
      | 2. 发送 "commit" IPC |
      v                      v
+-----+----------------------+--------+
|        Display Server              |
| - 管理窗口列表                     |
| - 读取 shm buffer                  |
| - 合成到 Framebuffer               |
+------------------------------------+
                |
                v
        +---------------+
        | Framebuffer   |
        | (物理屏幕)    |
        +---------------+
```

> 💡 **关键：Client 无法直接访问 Framebuffer，只能通过 Server 间接显示**。

---

## 🧱 二、共享内存（Shared Memory）机制

### 1. **系统调用：shmget / shmat**
```c
// 创建共享内存段
int sys_shmget(key_t key, size_t size, int shmflg) {
    // 1. 分配物理页
    void *pages = buddy_alloc_pages(ROUND_UP(size, PAGE_SIZE) / PAGE_SIZE);
    
    // 2. 创建共享内存对象
    struct shm_object *shm = kmem_cache_alloc(shm_cache);
    shm->key = key;
    shm->size = size;
    shm->paddr = (uint32_t)pages;
    shm->ref_count = 1;
    
    // 3. 加入全局哈希表
    hash_add(shm_table, key, shm);
    return key;
}

// 附加到进程地址空间
void *sys_shmat(int shmid, const void *shmaddr, int shmflg) {
    struct shm_object *shm = hash_find(shm_table, shmid);
    if (!shm) return MAP_FAILED;
    
    // 1. 找到未使用的虚拟地址
    uint32_t vaddr = find_free_vma(shm->size);
    
    // 2. 映射物理页到用户空间
    for (size_t i = 0; i < shm->size; i += PAGE_SIZE) {
        map_page(current_task->cr3, vaddr + i, 
                 shm->paddr + i, 
                 PAGE_PRESENT | PAGE_RW | PAGE_USER);
    }
    
    // 3. 增加引用计数
    shm->ref_count++;
    return (void*)vaddr;
}
```

### 2. **共享内存对象管理**
```c
struct shm_object {
    key_t key;
    size_t size;
    uint32_t paddr;         // 物理地址
    atomic_t ref_count;     // 引用计数
    spinlock_t lock;        // 同步锁
};
```

> 🔑 **共享内存的物理页在多个进程间共享，但虚拟地址可不同**。

---

## 🖼️ 三、Display Server 实现

### 1. **Server 初始化**
```c
// display_server.c
void display_server_init() {
    // 1. 创建命令通道（管道）
    display_ipc_fd = pipe_create();
    
    // 2. 创建共享内存池
    shm_pool = shmget(DISPLAY_SHM_KEY, 4 * 1024 * 1024, 0666); // 4MB
    
    // 3. 启动 Server 主循环（作为内核线程）
    kernel_thread_start(display_server_main, NULL);
}
```

### 2. **窗口管理**
```c
struct window {
    int id;
    int pid;                // 所属 Client PID
    int x, y, width, height;
    int shm_offset;         // 在共享内存中的偏移
    bool visible;
    struct window *next;
};

static struct window *window_list = NULL;
```

### 3. **Server 主循环**
```c
void display_server_main(void *arg) {
    while (1) {
        // 1. 读取 IPC 命令
        struct display_cmd cmd;
        if (read(display_ipc_fd, &cmd, sizeof(cmd)) <= 0) continue;
        
        // 2. 处理命令
        switch (cmd.type) {
            case CMD_CREATE_WINDOW:
                handle_create_window(&cmd);
                break;
            case CMD_COMMIT:
                handle_commit(&cmd);
                break;
            case CMD_CLOSE_WINDOW:
                handle_close_window(&cmd);
                break;
        }
        
        // 3. 合成并上屏（每 16ms 一帧）
        if (need_repaint) {
            compose_and_blit();
            need_repaint = false;
        }
    }
}
```

### 4. **合成与上屏（Blit）**
```c
void compose_and_blit() {
    // 1. 清空 Framebuffer
    fb_clear();
    
    // 2. 遍历窗口，从共享内存复制内容
    for (struct window *win = window_list; win; win = win->next) {
        if (!win->visible) continue;
        
        // 获取共享内存虚拟地址
        void *shm_vaddr = get_shm_vaddr(shm_pool) + win->shm_offset;
        
        // 3. 复制到 Framebuffer（带窗口偏移）
        for (int y = 0; y < win->height; y++) {
            for (int x = 0; x < win->width; x++) {
                uint32_t pixel = ((uint32_t*)shm_vaddr)[y * win->width + x];
                if (pixel & 0xFF000000) { // 有 alpha
                    fb_putpixel(win->x + x, win->y + y, pixel);
                }
            }
        }
    }
}
```

> ✅ **Server 是唯一写 Framebuffer 的实体**！

---

## 🖌️ 四、Client 绘图 API

### 1. **Client 初始化**
```c
// user/display_client.c
display_t* display_connect() {
    display_t *disp = malloc(sizeof(display_t));
    
    // 1. 连接到 Server IPC 通道
    disp->ipc_fd = open("/dev/display", O_RDWR);
    
    // 2. 附加共享内存
    disp->shm = shmat(DISPLAY_SHM_KEY, NULL, 0);
    disp->shm_size = 4 * 1024 * 1024;
    
    return disp;
}
```

### 2. **创建窗口**
```c
int display_create_window(display_t *disp, int width, int height) {
    struct display_cmd cmd = {
        .type = CMD_CREATE_WINDOW,
        .width = width,
        .height = height
    };
    write(disp->ipc_fd, &cmd, sizeof(cmd));
    
    // Server 返回窗口 ID
    int win_id;
    read(disp->ipc_fd, &win_id, sizeof(win_id));
    return win_id;
}
```

### 3. **绘图与提交**
```c
void* display_get_buffer(display_t *disp, int win_id) {
    // 从共享内存池分配区域（简化：固定偏移）
    return (char*)disp->shm + win_id * MAX_WINDOW_SIZE;
}

void display_commit(display_t *disp, int win_id) {
    struct display_cmd cmd = {
        .type = CMD_COMMIT,
        .win_id = win_id
    };
    write(disp->ipc_fd, &cmd, sizeof(cmd));
}
```

### 4. **Client 绘图示例**
```c
// user/painter.c
void _start() {
    display_t *disp = display_connect();
    int win = display_create_window(disp, 320, 240);
    
    uint32_t *buffer = display_get_buffer(disp, win);
    
    // 绘制红色矩形
    for (int y = 0; y < 100; y++) {
        for (int x = 0; x < 100; x++) {
            buffer[y * 320 + x] = 0xFFFF0000; // ARGB
        }
    }
    
    // 提交帧
    display_commit(disp, win);
    
    sleep(10); // 保持 10 秒
    exit(0);
}
```

---

## 📁 五、设备文件与 IPC 通道

### 1. **注册 Display 设备**
```c
// drivers/display.c
static struct file_operations display_fops = {
    .read = display_read,
    .write = display_write,
    .poll = display_poll, // 支持 select
};

void display_init() {
    struct device *dev = kmalloc(sizeof(struct device));
    strcpy(dev->name, "display");
    dev->devt = mkdev(MAJOR_DISPLAY, 0);
    device_register(dev);
    register_chrdev(MAJOR_DISPLAY, "display", &display_fops);
}
```

### 2. **IPC 通道实现**
- 使用 **匿名管道（pipe）** 或 **Unix 域套接字（简化版）**
- Server 创建管道，Client 通过 `/dev/display` 访问

---

## 🧪 六、测试：多 Client 合成显示

### 启动两个 Client：
1. **Terminal Client**：创建 640x480 窗口，显示文本
2. **Painter Client**：创建 320x240 窗口，绘制图形

### 运行效果：
- Screen 显示两个窗口：
  - 左上：Terminal（黑底白字）
  - 右下：Painter（红色矩形）
- 无画面撕裂，窗口独立更新

✅ **Display Server 成功合成多 Client 内容**！

---

## ⚠️ 七、优化方向

1. **双缓冲 Client**  
   - Client 有 front/back buffer，提交时交换
   - 避免 Server 读取到绘制一半的画面

2. **脏矩形更新**  
   - Client 只提交变化区域
   - 减少内存拷贝量

3. **硬件加速 Blit**  
   - 利用 GPU 的 BitBLT 指令
   - 需要更复杂的驱动

4. **输入事件路由**  
   - Server 接收键盘/鼠标事件
   - 根据窗口位置路由到 Client

> 💡 **现代 Wayland/X11 正是基于此架构演进而来**！

---

## 💬 写在最后

Display Server 是图形系统的**中枢神经**。  
它隔离了应用与硬件，  
通过共享内存实现高效通信，  
为现代 GUI 奠定基础。

今天你合成的第一个窗口，  
正是 Wayland、X11、Windows Desktop 的雏形。

> 🌟 **图形系统的优雅，在于让复杂对应用透明。**

---

📬 **动手挑战**：  
实现 `display_close_window`，并测试动态关闭窗口。  
欢迎在评论区分享你的多窗口截图！

👇 下一篇你想看：**输入事件处理与窗口焦点**，还是 **字体渲染与文本显示**？

---

**#操作系统 #内核开发 #DisplayServer #共享内存 #图形系统 #GUI #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“display”**，获取：
> - 完整 Display Server/Client 代码
> - 共享内存（shm）系统调用实现
> - 多窗口合成测试程序模板
