# 从零写 OS 内核-第三十一篇：画板工具与文字渲染 —— 在你的窗口中书写文字！

> **“图形界面不能只有像素，还需要文字！  
> 今天，我们实现画板工具，并集成文字渲染引擎，让标题栏、状态栏和画布都能显示文本！”**

在前几篇中，我们构建了完整的图形栈：  
✅ **Framebuffer 驱动**  
✅ **Display Server + Client 架构**  
✅ **窗口管理器（支持鼠标/键盘）**  

但所有内容都是**纯像素绘制**——  
窗口标题是硬编码字符串，  
画板工具无法添加文字标注。  

真正的图形系统需要**文字渲染能力**！  

今天，我们就来：  
✅ **实现位图字体渲染引擎**  
✅ **开发画板工具（Painter）**  
✅ **在标题栏/状态栏显示动态文本**  
✅ **支持画布上添加文字图层**  

让你的 OS 拥有**完整的文字显示能力**！

---

## 🔠 一、位图字体：最简文字渲染方案

### 为什么选择位图字体？
- **简单**：无需复杂字体解析（如 TrueType）  
- **高效**：直接内存拷贝，无浮点运算  
- **小巧**：8x16 字体仅需 128 字节/字符  

### 字体格式（8x16 monospace）：
- **ASCII 32-126**（可打印字符）  
- **每个字符 16 字节**（每行 1 字节，8 像素）  
- **1 = 像素开启，0 = 像素关闭**  

### 字体数据示例（'A'）：
```c
// font_8x16.h
static const uint8_t font_data[95][16] = {
    [0] = { // ' ' (space)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    },
    [1] = { // '!'
        0x00, 0x00, 0x00, 0x18, 0x18, 0x18, 0x18, 0x18,
        0x00, 0x00, 0x18, 0x18, 0x00, 0x00, 0x00, 0x00
    },
    // ... 
};
```

> 💡 **Linux 内核的 built-in font 采用相同格式**！

---

## 🖌️ 二、文字渲染引擎

### 1. **绘制单个字符**
```c
// user/font.c
void draw_char(uint32_t *fb, int fb_width, int x, int y, 
               char c, uint32_t color) {
    if (c < 32 || c > 126) return;
    
    const uint8_t *glyph = font_data[c - 32];
    
    for (int dy = 0; dy < 16; dy++) {
        uint8_t row = glyph[dy];
        for (int dx = 0; dx < 8; dx++) {
            if (row & (1 << (7 - dx))) { // 从左到右
                int px = x + dx;
                int py = y + dy;
                if (px >= 0 && px < fb_width && py >= 0 && py < SCREEN_HEIGHT) {
                    fb[py * fb_width + px] = color;
                }
            }
        }
    }
}
```

### 2. **绘制字符串**
```c
void draw_string(uint32_t *fb, int fb_width, int x, int y, 
                 const char *str, uint32_t color) {
    int cur_x = x;
    while (*str) {
        if (*str == '\n') {
            cur_x = x;
            y += 16;
        } else {
            draw_char(fb, fb_width, cur_x, y, *str, color);
            cur_x += 8;
        }
        str++;
    }
}
```

### 3. **测量字符串宽度**
```c
int string_width(const char *str) {
    int width = 0;
    while (*str) {
        if (*str != '\n') width += 8;
        str++;
    }
    return width;
}
```

> 🔑 **所有绘图函数操作 Framebuffer 直接内存**！

---

## 🎨 三、画板工具（Painter）实现

### 1. **Painter 窗口结构**
```c
// user/painter.c
struct painter_state {
    display_t *disp;
    uint32_t *canvas_buffer; // 画布内容（RGBA）
    int canvas_width, canvas_height;
    
    // 工具状态
    enum tool_type { TOOL_PEN, TOOL_ERASER, TOOL_TEXT } current_tool;
    uint32_t pen_color;
    int pen_size;
    
    // 文字输入状态
    bool entering_text;
    char text_input[256];
    int text_cursor;
    int text_x, text_y;
    
    // 状态栏信息
    char status_text[64];
};
```

### 2. **主循环**
```c
void painter_main() {
    struct painter_state state = {0};
    state.disp = display_connect();
    display_create_window(state.disp, 640, 480);
    state.canvas_width = state.disp->width;
    state.canvas_height = state.disp->height - 40; // 留出状态栏
    state.canvas_buffer = malloc(state.canvas_width * state.canvas_height * 4);
    memset(state.canvas_buffer, 0, state.canvas_width * state.canvas_height * 4);
    
    state.pen_color = 0xFF0000FF; // 蓝色
    state.current_tool = TOOL_PEN;
    strcpy(state.status_text, "Tool: Pen | Color: Blue");
    
    while (1) {
        // 处理事件（简化：通过 Display Server IPC）
        struct painter_event event;
        if (get_painter_event(&event)) {
            handle_painter_event(&state, &event);
        }
        
        // 重绘画板
        redraw_painter(&state);
        display_commit(state.disp);
    }
}
```

### 3. **绘制 Painter 窗口**
```c
void redraw_painter(struct painter_state *state) {
    uint32_t *fb = (uint32_t*)state->disp->buffer;
    int width = state->disp->width;
    
    // 1. 清屏（白色背景）
    for (int i = 0; i < width * state->disp->height; i++) {
        fb[i] = 0xFFFFFFFF;
    }
    
    // 2. 绘制画布内容
    for (int y = 0; y < state->canvas_height; y++) {
        for (int x = 0; x < state->canvas_width; x++) {
            uint32_t pixel = state->canvas_buffer[y * state->canvas_width + x];
            if (pixel & 0xFF000000) { // 有 alpha
                fb[(y + 30) * width + x] = pixel; // 标题栏下方 30px
            }
        }
    }
    
    // 3. 绘制标题栏
    draw_string(fb, width, 5, 5, "Painter - MyOS Drawing Tool", 0xFF000000);
    
    // 4. 绘制状态栏（底部）
    draw_string(fb, width, 5, state->disp->height - 15, 
                state->status_text, 0xFF000000);
    
    // 5. 绘制文字输入光标（如果正在输入）
    if (state->entering_text) {
        int cursor_x = state->text_x + string_width(state->text_input);
        draw_char(fb, width, cursor_x, state->text_y, '_', 0xFF000000);
    }
}
```

---

## ✍️ 四、文字工具实现

### 1. **切换到文字工具**
```c
void switch_to_text_tool(struct painter_state *state, int x, int y) {
    state->current_tool = TOOL_TEXT;
    state->text_x = x;
    state->text_y = y;
    state->text_input[0] = '\0';
    state->text_cursor = 0;
    state->entering_text = true;
    
    sprintf(state->status_text, "Text mode: Click to place, Enter to confirm");
}
```

### 2. **处理键盘输入**
```c
void handle_key_input(struct painter_state *state, char key) {
    if (!state->entering_text) return;
    
    if (key == '\n') {
        // 确认输入：将文字渲染到画布
        render_text_to_canvas(state);
        state->entering_text = false;
        strcpy(state->status_text, "Text added. Tool: Pen");
        state->current_tool = TOOL_PEN;
    } 
    else if (key == '\b') {
        // 退格
        if (state->text_cursor > 0) {
            state->text_cursor--;
            state->text_input[state->text_cursor] = '\0';
        }
    } 
    else if (state->text_cursor < 255 && key >= 32 && key <= 126) {
        // 可打印字符
        state->text_input[state->text_cursor] = key;
        state->text_cursor++;
        state->text_input[state->text_cursor] = '\0';
    }
}
```

### 3. **渲染文字到画布**
```c
void render_text_to_canvas(struct painter_state *state) {
    int x = state->text_x;
    int y = state->text_y - 30; // 转换为画布坐标
    
    // 逐字符绘制到 canvas_buffer
    int cur_x = x;
    for (int i = 0; state->text_input[i]; i++) {
        char c = state->text_input[i];
        const uint8_t *glyph = font_data[c - 32];
        
        for (int dy = 0; dy < 16; dy++) {
            uint8_t row = glyph[dy];
            for (int dx = 0; dx < 8; dx++) {
                if (row & (1 << (7 - dx))) {
                    int px = cur_x + dx;
                    int py = y + dy;
                    if (px >= 0 && px < state->canvas_width && 
                        py >= 0 && py < state->canvas_height) {
                        // 蓝色文字
                        state->canvas_buffer[py * state->canvas_width + px] = 0xFF0000FF;
                    }
                }
            }
        }
        cur_x += 8;
    }
}
```

---

## 🖱️ 五、鼠标与键盘集成

### 1. **Painter 事件处理**
```c
void handle_painter_event(struct painter_state *state, struct painter_event *event) {
    switch (event->type) {
        case EVENT_MOUSE_DOWN:
            if (state->current_tool == TOOL_TEXT) {
                // 在点击位置开始文字输入
                switch_to_text_tool(state, event->x, event->y);
            } else {
                // 画笔/橡皮擦
                paint_at(state, event->x, event->y, true);
            }
            break;
            
        case EVENT_MOUSE_MOVE:
            if (event->buttons & 1) { // 左键拖拽
                paint_at(state, event->x, event->y, false);
            }
            break;
            
        case EVENT_KEY_PRESS:
            handle_key_input(state, event->key);
            break;
    }
}
```

### 2. **Display Server 事件路由**
- Painter 向 Display Server 注册为**焦点窗口**  
- Display Server 将**鼠标/键盘事件**转发给 Painter  
- Painter 通过 `draw_string` 更新状态栏  

---

## 📁 六、字体与资源部署

### 1. **字体编译进二进制**
```c
// user/font_8x16.h (由脚本生成)
static const uint8_t font_data[95][16] = {
    #include "font_8x16_data.h"
};
```

### 2. **Painter 应用部署**
```
/bin/painter          # 画板可执行文件
/usr/share/fonts/     # 字体目录（可选）
```

### 3. **.desktop 文件**
```ini
[Desktop Entry]
Name=Painter
Exec=/bin/painter
Icon=/usr/share/icons/painter.png
Type=Application
Categories=Graphics;
```

---

## 🧪 七、测试：文字渲染效果

### 操作流程：
1. **启动 Painter** → 显示标题栏和状态栏  
2. **点击画布** → 用蓝色画笔绘制  
3. **切换文字工具** → 点击画布位置  
4. **输入 "Hello MyOS!"** → 显示输入光标  
5. **按 Enter** → 文字渲染到画布  
6. **状态栏实时更新**工具状态  

### 运行效果：
- 标题栏：`"Painter - MyOS Drawing Tool"`  
- 状态栏：`"Tool: Text | Click to place"`  
- 画布：蓝色手绘线条 + 蓝色文字 "Hello MyOS!"  
- 文字输入时显示闪烁光标（简化版：静态下划线）  

✅ **文字渲染与画板工具完美集成**！

---

## ⚙️ 八、扩展方向

1. **字体抗锯齿**  
   - 使用灰度字体实现平滑边缘  
   - 需要 alpha 混合计算

2. **多字体支持**  
   - 加载外部字体文件（如 .bdf）  
   - 支持不同字号

3. **文本格式化**  
   - 粗体/斜体（通过位图变换）  
   - 对齐方式（居中、右对齐）

4. **Unicode 支持**  
   - UTF-8 解码  
   - 支持中文（需大字体文件）

> 💡 **FreeType 是工业级字体渲染库，但位图字体是起点**！

---

## 💬 写在最后

文字渲染是图形系统的**点睛之笔**。  
它将像素转化为信息，  
让界面真正具备沟通能力。

今天你绘制的第一个 "Hello MyOS!"，  
正是无数 GUI 文本显示的起点。

> 🌟 **文字是思想的载体，渲染是技术的艺术。**

---

📬 **动手挑战**：  
添加字体颜色选择器，并实现文字居中对齐功能。  
欢迎在评论区分享你的画板作品截图！

👇 下一篇你想看：**任务栏（Taskbar）实现**，还是 **多语言输入法框架**？

---

**#操作系统 #内核开发 #文字渲染 #画板工具 #字体 #图形界面 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“painter”**，获取：
> - 完整 Painter 画板工具源码
> - 8x16 位图字体数据
> - 文字渲染引擎模板
