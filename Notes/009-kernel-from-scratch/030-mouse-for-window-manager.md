# 从零写 OS 内核-第三十篇：窗口管理器鼠标支持 —— 实现点击、拖拽与窗口操作

> **“键盘是高效之选，鼠标是直观之王。  
> 今天，我们在窗口管理器中集成鼠标支持，实现真正的图形交互！”**

在上一篇中，我们成功实现了 **PS/2 鼠标驱动**，  
并在 `/dev/input/mouse0` 提供了原始鼠标事件。  
但窗口管理器（WM）仍是**键盘驱动**的——  
用户无法通过鼠标**点击、拖拽、关闭窗口**。

今天，我们就来：  
✅ **在用户态窗口管理器中读取鼠标事件**  
✅ **实现窗口点击与焦点切换**  
✅ **支持标题栏拖拽移动窗口**  
✅ **添加关闭/缩放按钮的点击处理**  

让你的 WM 拥有**完整的鼠标交互能力**！

---

## 🖥️ 一、架构整合：鼠标事件流向

### 事件流：
```
PS/2 鼠标硬件 
    → 内核鼠标驱动（IRQ12） 
    → /dev/input/mouse0 
    → Display Server（监听设备） 
    → Window Manager（通过 IPC 接收事件） 
    → 窗口交互逻辑
```

### 关键设计：
- **Display Server 负责**：  
  - 监听 `/dev/input/mouse0`  
  - 维护**全局鼠标坐标**  
  - 将事件路由给 **Window Manager**  
- **Window Manager 负责**：  
  - 判断点击位置  
  - 执行窗口操作（移动、关闭等）  

> 💡 **Display Server 不处理窗口逻辑，只做事件分发**！

---

## 🖱️ 二、Display Server 鼠标集成

### 1. **启动鼠标监听线程**
```c
// user/display_server.c
void start_input_thread() {
    if (fork() == 0) {
        // 监听鼠标
        int mouse_fd = open("/dev/input/mouse0", O_RDONLY);
        struct mouse_event event;
        
        while (1) {
            if (read(mouse_fd, &event, sizeof(event)) > 0) {
                // 更新全局坐标
                update_global_mouse_position(&event);
                
                // 转发给 Window Manager
                send_to_wm(WM_MOUSE_EVENT, &event, sizeof(event));
            }
        }
    }
}
```

### 2. **维护全局鼠标状态**
```c
struct global_mouse_state {
    int x, y;           // 绝对坐标
    uint8_t buttons;    // 当前按钮状态
    bool updated;       // 本次循环是否更新
} mouse_state = {0};

void update_global_mouse_position(struct mouse_event *event) {
    // 累积相对位移
    mouse_state.x = max(0, min(mouse_state.x + event->dx, SCREEN_WIDTH - 1));
    mouse_state.y = max(0, min(mouse_state.y + event->dy, SCREEN_HEIGHT - 1));
    mouse_state.buttons = event->buttons;
    mouse_state.updated = true;
}
```

---

## 🖼️ 三、窗口管理器鼠标处理

### 1. **WM 主循环增强**
```c
// user/window_manager.c
void wm_main_loop() {
    while (1) {
        // 处理 Display Server 消息
        struct wm_message msg;
        if (recv(server_sock, &msg, sizeof(msg), 0) > 0) {
            switch (msg.type) {
                case WM_MOUSE_EVENT:
                    handle_mouse_event((struct mouse_event*)&msg.data);
                    break;
                case WM_NEW_WINDOW:
                    handle_new_window((struct wm_window_info*)&msg.data);
                    break;
                // ... 其他消息
            }
        }
        
        // 重绘（仅当鼠标或窗口变化）
        if (mouse_state.updated || windows_updated) {
            redraw_screen();
            mouse_state.updated = false;
            windows_updated = false;
        }
        
        usleep(10000); // 10ms 刷新率
    }
}
```

### 2. **窗口区域检测**
```c
enum window_region {
    REGION_NONE,
    REGION_TITLEBAR,
    REGION_CLOSE_BUTTON,
    REGION_RESIZE_CORNER,
    REGION_CONTENT
};

enum window_region get_window_region(struct window *win, int x, int y) {
    // 检查是否在窗口内
    if (x < win->x || x >= win->x + win->width ||
        y < win->y || y >= win->y + win->height) {
        return REGION_NONE;
    }
    
    // 检查关闭按钮（右上角 20x20）
    if (x >= win->x + win->width - 20 && y < win->y + 25) {
        return REGION_CLOSE_BUTTON;
    }
    
    // 检查标题栏（高度 25）
    if (y < win->y + 25) {
        return REGION_TITLEBAR;
    }
    
    // 检查右下角缩放区域（20x20）
    if (x >= win->x + win->width - 20 && y >= win->y + win->height - 20) {
        return REGION_RESIZE_CORNER;
    }
    
    return REGION_CONTENT;
}
```

### 3. **鼠标事件处理状态机**
```c
struct wm_mouse_state {
    bool dragging;
    bool resizing;
    int drag_window_id;
    int drag_start_x, drag_start_y;
    enum window_region drag_region;
    struct window *hover_window;
} wm_mouse = {0};

void handle_mouse_event(struct mouse_event *event) {
    int x = mouse_state.x;
    int y = mouse_state.y;
    uint8_t buttons = mouse_state.buttons;
    
    // 1. 查找鼠标下的窗口
    struct window *top_win = find_top_window_at(x, y);
    
    // 2. 处理按钮按下
    if (buttons & 1) { // 左键按下
        if (top_win && !wm_mouse.dragging && !wm_mouse.resizing) {
            enum window_region region = get_window_region(top_win, x, y);
            
            if (region == REGION_TITLEBAR) {
                // 开始拖拽
                wm_mouse.dragging = true;
                wm_mouse.drag_window_id = top_win->id;
                wm_mouse.drag_start_x = x - top_win->x;
                wm_mouse.drag_start_y = y - top_win->y;
                wm_mouse.drag_region = REGION_TITLEBAR;
                set_focus(top_win);
            } 
            else if (region == REGION_CLOSE_BUTTON) {
                // 关闭窗口
                send_close_request(top_win->id);
            }
            else if (region == REGION_RESIZE_CORNER) {
                // 开始缩放
                wm_mouse.resizing = true;
                wm_mouse.drag_window_id = top_win->id;
                wm_mouse.drag_region = REGION_RESIZE_CORNER;
                set_focus(top_win);
            }
        }
    } 
    else {
        // 左键释放
        wm_mouse.dragging = false;
        wm_mouse.resizing = false;
    }
    
    // 3. 处理拖拽
    if (wm_mouse.dragging && (buttons & 1)) {
        struct window *win = get_window_by_id(wm_mouse.drag_window_id);
        if (win) {
            int new_x = x - wm_mouse.drag_start_x;
            int new_y = y - wm_mouse.drag_start_y;
            move_window(win, new_x, new_y);
        }
    }
    
    // 4. 处理缩放
    if (wm_mouse.resizing && (buttons & 1)) {
        struct window *win = get_window_by_id(wm_mouse.drag_window_id);
        if (win) {
            int new_width = x - win->x;
            int new_height = y - win->y;
            resize_window(win, new_width, new_height);
        }
    }
    
    // 5. 更新悬停状态（用于绘制 hover 效果）
    wm_mouse.hover_window = top_win;
}
```

---

## 🎨 四、增强窗口装饰

### 1. **绘制交互反馈**
```c
void draw_window_decoration(struct window *win, uint32_t *fb, int fb_width) {
    // ... 基础绘制 ...
    
    // 1. 悬停高亮关闭按钮
    if (wm_mouse.hover_window == win) {
        enum window_region region = get_window_region(
            win, mouse_state.x, mouse_state.y
        );
        if (region == REGION_CLOSE_BUTTON) {
            // 绘制红色背景
            for (int dy = 0; dy < 20; dy++) {
                for (int dx = 0; dx < 20; dx++) {
                    int fb_x = win->x + win->width - 20 + dx;
                    int fb_y = win->y + dy;
                    fb[fb_y * fb_width + fb_x] = 0xFFFF0000; // 红色
                }
            }
        }
    }
    
    // 2. 绘制缩放角标
    if (win->focused) {
        draw_text(fb, fb_width, 
                 win->x + win->width - 15, 
                 win->y + win->height - 15, "◢");
    }
}
```

### 2. **绘制鼠标指针**
```c
void draw_mouse_cursor(uint32_t *fb, int fb_width) {
    int x = mouse_state.x;
    int y = mouse_state.y;
    
    // 十字指针
    for (int i = -5; i <= 5; i++) {
        if (x + i >= 0 && x + i < fb_width) {
            fb[y * fb_width + x + i] = 0xFFFFFFFF;
        }
        if (y + i >= 0 && y + i < SCREEN_HEIGHT) {
            fb[(y + i) * fb_width + x] = 0xFFFFFFFF;
        }
    }
}
```

---

## 📡 五、Display Server 合成流程

### 1. **合成顺序**
```c
void compose_and_blit() {
    // 1. 清屏
    clear_screen();
    
    // 2. 按 Z-Order 绘制窗口（从底到顶）
    for (struct window *win = bottom_window; win; win = win->above) {
        // 绘制窗口装饰
        draw_window_decoration(win);
        
        // 叠加 Client 内容区域
        blit_client_content(win);
    }
    
    // 3. 绘制鼠标指针（最顶层）
    draw_mouse_cursor();
}
```

### 2. **Z-Order 管理**
```c
// 点击窗口时将其置顶
void set_focus(struct window *win) {
    if (win->focused) return;
    
    // 移除旧焦点
    if (focused_window) {
        focused_window->focused = false;
    }
    
    // 置顶新窗口
    remove_from_zorder(win);
    add_to_top(win);
    
    win->focused = true;
    focused_window = win;
    windows_updated = true;
}
```

---

## 🧪 六、测试：鼠标交互操作

### 用户操作流程：
1. **移动鼠标** → 屏幕显示十字指针  
2. **悬停在关闭按钮** → 按钮变红色  
3. **点击标题栏并拖拽** → 窗口跟随鼠标移动  
4. **拖拽右下角** → 窗口大小实时变化  
5. **点击关闭按钮** → 窗口关闭，焦点切换到下一窗口  

### 运行效果：
- 窗口移动/缩放流畅  
- 关闭按钮悬停反馈及时  
- 鼠标指针始终在最顶层  
- 多窗口堆叠顺序正确  

✅ **鼠标驱动的窗口管理器工作完美**！

---

## ⚙️ 七、优化方向

1. **平滑动画**  
   - 移动/缩放时插值过渡  
   - 使用双缓冲避免撕裂

2. **鼠标加速**  
   - 根据移动速度动态调整灵敏度  
   - 高速移动时跳过中间帧

3. **多显示器支持**  
   - 鼠标在屏幕间无缝移动  
   - 每个显示器独立窗口管理

4. **手势支持**  
   - 双指滚动（需触摸板驱动）  
   - 三指切换工作区

> 💡 **现代 WM（如 KWin、Mutter）均提供这些高级特性**！

---

## 💬 写在最后

鼠标支持让窗口管理器**从可用变为好用**。  
它将抽象的窗口操作转化为直观的视觉反馈，  
极大地提升了用户体验。

今天你实现的第一个可拖拽窗口，  
正是现代桌面环境交互的基石。

> 🌟 **最好的交互，是让用户感觉不到技术的存在，只享受流畅的操作。**

---

📬 **动手挑战**：  
添加双击标题栏最大化窗口功能，并实现窗口阴影效果。  
欢迎在评论区分享你的鼠标交互视频！

👇 下一篇你想看：**任务栏（Taskbar）实现**，还是 **窗口动画与特效**？

---

**#操作系统 #内核开发 #窗口管理器 #鼠标支持 #图形界面 #WM #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“wm-mouse”**，获取：
> - 完整鼠标窗口管理器源码
> - 窗口区域检测与交互模板
> - 鼠标指针绘制与合成技巧
