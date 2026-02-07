# 从零写 OS 内核-第二十八篇：窗口管理器（Window Manager） —— 通过键盘实现窗口控制

> **“没有鼠标？没关系！真正的极客用键盘掌控一切。  
> 今天，我们实现窗口管理器（Window Manager），通过键盘快捷键移动、缩放窗口！”**

在前几篇中，我们构建了用户态 Display Server、Launcher 和多个应用，  
但窗口仍是**静态的**：  
- 位置固定  
- 大小不可变  
- 无标题栏、边框等装饰  

虽然我们尚未实现鼠标驱动，但**键盘足以控制窗口**！  
Unix 哲学告诉我们：**键盘是终极交互设备**。

今天，我们就来：  
✅ **设计窗口管理协议**  
✅ **实现用户态窗口管理器**  
✅ **通过键盘快捷键控制窗口**  
✅ **绘制窗口装饰（标题栏、边框）**  

让你的 OS 拥有**键盘驱动的窗口管理能力**！

---

## 🖥️ 一、窗口管理器架构（无鼠标版）

### 核心思想：
- **Display Server**：只负责**合成与上屏**  
- **Window Manager**：作为**特殊 Client**，管理所有窗口的装饰与键盘交互  
- **普通 Client**：只绘制**内容区域**  

### 交互方式：
| 操作 | 快捷键 |
|------|--------|
| **切换焦点窗口** | `Alt+Tab` |
| **移动窗口** | `Alt+方向键` |
| **缩放窗口** | `Alt+Shift+方向键` |
| **关闭窗口** | `Alt+F4` |
| **最大化** | `Alt+F10` |

> 💡 **i3、xmonad 等平铺式窗口管理器正是以键盘为中心**！

---

## 📐 二、窗口管理协议扩展

### 1. **新命令类型**
```c
// display_protocol.h
enum display_cmd_type {
    CMD_CREATE_WINDOW,
    CMD_COMMIT,
    CMD_CLOSE_WINDOW,
    // WM 新增
    CMD_WM_NEW_WINDOW,      // Server 通知 WM 新窗口
    CMD_WM_FOCUS_REQUEST,   // WM 请求焦点切换
    CMD_WM_MOVE_REQUEST,    // WM 请求移动窗口
    CMD_WM_RESIZE_REQUEST,  // WM 请求缩放窗口
    CMD_WM_CLOSE_REQUEST,   // WM 请求关闭窗口
};
```

### 2. **窗口状态**
```c
struct window {
    int id;
    int client_pid;
    int x, y;               // 窗口位置（含装饰）
    int width, height;      // 窗口大小（含装饰）
    int content_x, content_y; // 内容区域偏移
    int content_width, content_height;
    bool focused;
    char title[64];
    struct window *next;
};
```

---

## ⌨️ 三、键盘事件处理

### 1. **Display Server 添加键盘事件通道**
- TTY 驱动将特殊按键（如 `Alt`）发送给 **Window Manager**  
- 普通字符仍发送给**焦点窗口**

### 2. **WM 键盘状态机**
```c
struct wm_state {
    bool alt_pressed;
    bool shift_pressed;
    struct window *focused_window;
    struct window *windows;
};
```

### 3. **处理按键事件**
```c
void handle_key_event(uint8_t scancode, bool pressed) {
    // 处理修饰键
    if (scancode == SCANCODE_ALT) {
        wm_state.alt_pressed = pressed;
        return;
    }
    if (scancode == SCANCODE_SHIFT) {
        wm_state.shift_pressed = pressed;
        return;
    }
    
    // 仅处理 Alt+ 组合键
    if (!wm_state.alt_pressed || !pressed) {
        // 发送给焦点窗口
        if (wm_state.focused_window) {
            send_key_to_client(wm_state.focused_window->client_pid, scancode);
        }
        return;
    }
    
    // 处理快捷键
    if (scancode == SCANCODE_TAB) {
        switch_focus_window();
    } else if (scancode == SCANCODE_F4) {
        close_focused_window();
    } else if (scancode == SCANCODE_F10) {
        maximize_focused_window();
    } else if (scancode == SCANCODE_LEFT) {
        if (wm_state.shift_pressed) {
            resize_focused_window(-10, 0);
        } else {
            move_focused_window(-10, 0);
        }
    } else if (scancode == SCANCODE_RIGHT) {
        if (wm_state.shift_pressed) {
            resize_focused_window(10, 0);
        } else {
            move_focused_window(10, 0);
        }
    }
    // ... 其他方向键
}
```

---

## 🖼️ 四、用户态窗口管理器实现

### 1. **WM 主循环**
```c
// user/window_manager.c
void _start() {
    // 1. 连接到 Display Server
    display_t *disp = display_connect();
    
    // 2. 注册为 Window Manager
    register_as_window_manager(disp);
    
    // 3. 主循环
    while (1) {
        // 处理 Display Server 消息
        struct display_cmd cmd;
        if (recv(disp->sock_fd, &cmd, sizeof(cmd), 0) > 0) {
            handle_server_message(&cmd);
        }
        
        // 处理键盘事件（通过特殊 IPC 通道）
        uint8_t scancode;
        bool pressed;
        if (get_keyboard_event(&scancode, &pressed)) {
            handle_key_event(scancode, pressed);
        }
        
        // 重绘装饰
        if (needs_repaint) {
            redraw_decorations(disp);
            needs_repaint = false;
        }
    }
}
```

### 2. **处理新窗口**
```c
void handle_server_message(struct display_cmd *cmd) {
    if (cmd->type == CMD_WM_NEW_WINDOW) {
        struct window *win = create_window(
            cmd->win_id, cmd->client_pid,
            cmd->x, cmd->y, cmd->width, cmd->height
        );
        
        // 设置默认标题
        strcpy(win->title, "Application");
        
        // 首个窗口自动获得焦点
        if (!wm_state.focused_window) {
            set_focus(win);
        }
        
        // 通知 Client 内容区域
        struct display_cmd resize_cmd = {
            .type = CMD_WM_RESIZE_REQUEST,
            .win_id = cmd->win_id,
            .x = win->content_x,
            .y = win->content_y,
            .width = win->content_width,
            .height = win->content_height
        };
        send_to_client(cmd->client_pid, &resize_cmd);
    }
}
```

### 3. **绘制窗口装饰**
```c
void draw_window_decoration(struct window *win, uint32_t *fb, int fb_width) {
    int x = win->x, y = win->y;
    int w = win->width, h = win->height;
    
    // 标题栏背景（聚焦=蓝色，非聚焦=灰色）
    uint32_t title_bg = win->focused ? 0xFF3366CC : 0xFFCCCCCC;
    for (int dy = 0; dy < 25; dy++) {
        for (int dx = 0; dx < w; dx++) {
            fb[(y + dy) * fb_width + (x + dx)] = title_bg;
        }
    }
    
    // 标题文字
    draw_text(fb, fb_width, x + 5, y + 5, win->title);
    
    // 关闭提示
    draw_text(fb, fb_width, x + w - 80, y + 5, "Alt+F4 to close");
    
    // 边框
    uint32_t border = 0xFF888888;
    for (int i = 0; i < w; i++) {
        fb[(y + 25) * fb_width + (x + i)] = border;
        fb[(y + h - 1) * fb_width + (x + i)] = border;
    }
    for (int i = 0; i < h; i++) {
        fb[(y + i) * fb_width + (x)] = border;
        fb[(y + i) * fb_width + (x + w - 1)] = border;
    }
}
```

### 4. **窗口操作函数**
```c
void move_focused_window(int dx, int dy) {
    if (!wm_state.focused_window) return;
    
    struct window *win = wm_state.focused_window;
    win->x += dx;
    win->y += dy;
    
    // 通知 Client 新位置
    struct display_cmd cmd = {
        .type = CMD_WM_MOVE_REQUEST,
        .win_id = win->id,
        .x = win->x,
        .y = win->y
    };
    send_to_client(win->client_pid, &cmd);
    
    needs_repaint = true;
}

void resize_focused_window(int dw, int dh) {
    if (!wm_state.focused_window) return;
    
    struct window *win = wm_state.focused_window;
    win->width = max(100, win->width + dw);
    win->height = max(80, win->height + dh);
    
    // 重新计算内容区域
    win->content_width = win->width - 10;
    win->content_height = win->height - 35;
    
    // 通知 Client
    struct display_cmd cmd = {
        .type = CMD_WM_RESIZE_REQUEST,
        .win_id = win->id,
        .width = win->content_width,
        .height = win->content_height
    };
    send_to_client(win->client_pid, &cmd);
    
    needs_repaint = true;
}

void close_focused_window() {
    if (!wm_state.focused_window) return;
    
    struct display_cmd cmd = {
        .type = CMD_WM_CLOSE_REQUEST,
        .win_id = wm_state.focused_window->id
    };
    send_to_client(wm_state.focused_window->client_pid, &cmd);
}
```

---

## 📡 五、Client 端适配

### 1. **处理 WM 命令**
```c
// user/client.c
void handle_wm_command(struct display_cmd *cmd) {
    switch (cmd->type) {
        case CMD_WM_MOVE_REQUEST:
            // 更新窗口位置（仅装饰，内容区域不变）
            break;
        case CMD_WM_RESIZE_REQUEST:
            // 重新分配共享内存缓冲区
            if (shm_buffer) {
                shmdt(shm_buffer);
            }
            shm_fd = shmget(IPC_PRIVATE, cmd->width * cmd->height * 4, 0666);
            shm_buffer = shmat(shm_fd, NULL, 0);
            
            // 重新传递 shm_fd 给 WM
            send_shm_to_wm(shm_fd);
            break;
        case CMD_WM_CLOSE_REQUEST:
            exit(0);
            break;
    }
}
```

### 2. **绘图区域自动适配**
- Client 始终绘制到 **内容区域缓冲区**  
- WM 负责将内容区域合成到正确位置

---

## 🧪 六、测试：键盘控制窗口

### 操作流程：
1. **启动 Terminal 和 Painter**  
2. **Alt+Tab** → 切换焦点窗口（标题栏颜色变化）  
3. **Alt+→** → 焦点窗口向右移动 10 像素  
4. **Alt+Shift+→** → 焦点窗口宽度增加 10 像素  
5. **Alt+F4** → 关闭焦点窗口  

### 运行效果：
- 焦点窗口标题栏：蓝色  
- 非焦点窗口标题栏：灰色  
- 窗口移动/缩放实时生效  
- 关闭窗口后自动切换到下一个窗口  

✅ **键盘驱动的窗口管理器工作正常**！

---

## ⚙️ 七、优势与扩展

### 优势：
- **高效**：键盘操作比鼠标更快（对程序员友好）  
- **可靠**：无需复杂鼠标驱动  
- **可脚本化**：快捷键可绑定到脚本  

### 扩展方向：
1. **窗口布局管理**  
   - 平铺布局（i3 风格）  
   - 标签式布局（一个窗口区域，多个标签页）

2. **工作区（Workspace）**  
   - `Ctrl+Alt+1/2/3` 切换工作区  
   - 每个工作区独立窗口集合

3. **窗口规则**  
   - 根据应用名称自动放置窗口  
   - 自动最大化特定应用

> 💡 **键盘为中心的设计，正是高效工作流的核心**！

---

## 💬 写在最后

窗口管理器不需要鼠标，  
键盘足以掌控整个桌面。  
在命令行与快捷键的世界里，  
效率与优雅并存。

今天你实现的第一个键盘窗口管理器，  
正是 i3、xmonad 等现代 WM 的简化版。

> 🌟 **真正的掌控感，来自指尖而非鼠标。**

---

📬 **动手挑战**：  
添加 `Alt+F11` 全屏快捷键，并实现工作区切换（Alt+1/2/3）。  
欢迎在评论区分享你的键盘快捷键配置！

👇 下一篇你想看：**任务栏（Taskbar）实现**，还是 **终端多标签支持**？

---

**#操作系统 #内核开发 #窗口管理器 #KeyboardDriven #WM #图形界面 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“wm-keyboard”**，获取：
> - 完整键盘窗口管理器源码
> - 窗口装饰绘制模板
> - 快捷键配置表
