
# 从零写 OS 内核-第二十七篇：应用启动器（Launcher） —— 构建你的桌面环境雏形

> **“图形界面不能只有窗口，还需要一个入口！  
> 今天，我们实现应用启动器（Launcher），通过 .desktop 文件管理应用，点击即启动！”**

在上一篇中，我们成功将 Display Server 与 Client 移至用户态，  
实现了多窗口图形系统。  
但用户仍需在终端手动输入 `exec("/bin/painter")` 启动程序——  
**缺少一个直观的应用入口**。

真正的桌面环境需要 **应用启动器（Launcher）**：  
- **图标化显示应用**  
- **通过配置文件定义应用属性**  
- **点击图标启动程序**  

今天，我们就来：  
✅ **设计 .desktop 配置文件格式**  
✅ **实现 Launcher 应用**  
✅ **开发两个示例应用（Terminal & Painter）**  
✅ **集成图标显示与点击启动**  

让你的 OS 拥有**完整的桌面体验雏形**！

---

## 🖥️ 一、.desktop 文件：应用的元数据标准

我们采用简化版 **FreeDesktop.org Desktop Entry Specification**：

### 文件格式（INI 风格）：
```ini
[Desktop Entry]
Name=Terminal
Comment=Command line interface
Exec=/bin/terminal
Icon=/usr/share/icons/terminal.png
Type=Application
Categories=System;
```

### 关键字段：
| 字段 | 说明 |
|------|------|
| **Name** | 应用显示名称 |
| **Exec** | 可执行文件路径 |
| **Icon** | 图标路径（PNG 格式）|
| **Type** | 固定为 "Application" |
| **Categories** | 应用分类（可选）|

> 💡 **.desktop 文件通常放在 `/usr/share/applications/`**

---

## 🚀 二、Launcher 应用设计

### 核心功能：
1. **扫描 `/usr/share/applications/` 目录**  
2. **解析 .desktop 文件**  
3. **加载图标并显示为按钮**  
4. **点击按钮启动对应应用**

### Launcher 界面布局：
```
+----------------------------------+
| [Terminal Icon]  [Painter Icon]  |
| Terminal         Painter         |
+----------------------------------+
```

---

## 🧩 三、实现 .desktop 解析器

### 1. **Desktop Entry 结构**
```c
// user/launcher/desktop.h
typedef struct {
    char name[64];
    char exec[128];
    char icon[128];
    char categories[64];
} desktop_entry_t;
```

### 2. **INI 解析函数**
```c
// user/launcher/desktop.c
int parse_desktop_file(const char *path, desktop_entry_t *entry) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    
    char buffer[1024];
    int n = read(fd, buffer, sizeof(buffer) - 1);
    close(fd);
    buffer[n] = '\0';
    
    // 查找 [Desktop Entry] 段
    char *section = strstr(buffer, "[Desktop Entry]");
    if (!section) return -1;
    
    // 解析关键字段
    parse_key_value(section, "Name=", entry->name, sizeof(entry->name));
    parse_key_value(section, "Exec=", entry->exec, sizeof(entry->exec));
    parse_key_value(section, "Icon=", entry->icon, sizeof(entry->icon));
    parse_key_value(section, "Categories=", entry->categories, sizeof(entry->categories));
    
    return 0;
}

void parse_key_value(char *section, const char *key, char *value, size_t max_len) {
    char *line = strstr(section, key);
    if (!line) return;
    
    line += strlen(key);
    char *end = strchr(line, '\n');
    if (end) *end = '\0';
    
    strncpy(value, line, max_len - 1);
    value[max_len - 1] = '\0';
}
```

---

## 🖼️ 四、PNG 图标加载（简化版）

为简化，我们假设图标是 **32x32 RGBA PNG**，并使用**简化解码器**：

### 1. **PNG 头部结构**
```c
typedef struct {
    uint8_t signature[8];   // 必须为 137 80 78 71 13 10 26 10
    uint32_t width;
    uint32_t height;
    uint8_t bit_depth;
    uint8_t color_type;     // 6 = RGBA
    // ... 其他字段忽略
} png_header_t;
```

### 2. **加载图标到内存**
```c
uint32_t* load_png_icon(const char *path, int *width, int *height) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    
    // 读取头部
    png_header_t header;
    read(fd, &header, sizeof(header));
    
    // 验证 PNG 签名
    if (memcmp(header.signature, "\x89PNG\r\n\x1a\n", 8) != 0) {
        close(fd);
        return NULL;
    }
    
    *width = __builtin_bswap32(header.width);
    *height = __builtin_bswap32(header.height);
    
    if (header.color_type != 6 || *width != 32 || *height != 32) {
        close(fd);
        return NULL;
    }
    
    // 跳过其他 chunk，直接读取 IDAT（简化：假设无压缩）
    lseek(fd, 33, SEEK_SET); // 跳过头部和 IHDR
    
    uint32_t *pixels = malloc(32 * 32 * 4);
    read(fd, pixels, 32 * 32 * 4);
    close(fd);
    
    return pixels;
}
```

> ⚠️ **实际 PNG 需解压缩（zlib），此处为演示简化**。

---

## 🖱️ 五、Launcher 主程序

### 1. **扫描应用目录**
```c
// user/launcher/launcher.c
#define MAX_APPS 16
desktop_entry_t apps[MAX_APPS];
uint32_t *app_icons[MAX_APPS];
int num_apps = 0;

void scan_applications() {
    int dir_fd = open("/usr/share/applications", O_RDONLY);
    if (dir_fd < 0) return;
    
    struct dirent entry;
    while (readdir(dir_fd, &entry) > 0) {
        if (strstr(entry.d_name, ".desktop")) {
            char path[256];
            sprintf(path, "/usr/share/applications/%s", entry.d_name);
            
            if (num_apps < MAX_APPS) {
                if (parse_desktop_file(path, &apps[num_apps]) == 0) {
                    // 加载图标
                    app_icons[num_apps] = load_png_icon(apps[num_apps].icon, NULL, NULL);
                    num_apps++;
                }
            }
        }
    }
    close(dir_fd);
}
```

### 2. **绘制 Launcher 界面**
```c
void draw_launcher(display_t *disp) {
    uint32_t *buffer = (uint32_t*)disp->buffer;
    
    // 清屏（浅灰色）
    for (int i = 0; i < disp->width * disp->height; i++) {
        buffer[i] = 0xFFE0E0E0;
    }
    
    // 绘制应用图标
    for (int i = 0; i < num_apps; i++) {
        int x = 50 + i * 100;
        int y = 50;
        
        // 绘制图标
        if (app_icons[i]) {
            for (int dy = 0; dy < 32; dy++) {
                for (int dx = 0; dx < 32; dx++) {
                    int src_idx = dy * 32 + dx;
                    int dst_idx = (y + dy) * disp->width + (x + dx);
                    if (dst_idx < disp->width * disp->height) {
                        buffer[dst_idx] = app_icons[i][src_idx];
                    }
                }
            }
        }
        
        // 绘制应用名称
        draw_text(buffer, disp->width, x, y + 40, apps[i].name);
    }
    
    display_commit(disp);
}
```

### 3. **处理点击事件（简化：通过终端输入模拟）**
```c
void handle_click(int app_index) {
    if (app_index >= 0 && app_index < num_apps) {
        // 启动应用
        if (fork() == 0) {
            char *args[] = {apps[app_index].exec, NULL};
            exec(apps[app_index].exec, args);
            exit(1);
        }
    }
}
```

> 💡 **完整版需集成鼠标事件，此处用终端输入模拟**。

---

## 📁 六、应用与配置文件部署

### 1. **Terminal 应用**
- **可执行文件**：`/bin/terminal`  
- **.desktop 文件**：`/usr/share/applications/terminal.desktop`
  ```ini
  [Desktop Entry]
  Name=Terminal
  Exec=/bin/terminal
  Icon=/usr/share/icons/terminal.png
  Type=Application
  Categories=System;
  ```

### 2. **Painter 应用**
- **可执行文件**：`/bin/painter`  
- **.desktop 文件**：`/usr/share/applications/painter.desktop`
  ```ini
  [Desktop Entry]
  Name=Painter
  Exec=/bin/painter
  Icon=/usr/share/icons/painter.png
  Type=Application
  Categories=Graphics;
  ```

### 3. **文件系统布局**
```
/usr/share/applications/
  ├── terminal.desktop
  └── painter.desktop
/usr/share/icons/
  ├── terminal.png
  └── painter.png
/bin/
  ├── launcher
  ├── terminal
  └── painter
```

---

## 🧪 七、运行效果

1. **启动系统** → `init` 启动 `launcher`  
2. **Launcher 扫描 .desktop 文件**  
3. **加载图标并显示应用按钮**  
4. **用户“点击” Terminal 按钮** → 启动 terminal 应用  
5. **Terminal 窗口出现在屏幕**  

终端模拟交互：
```
myos$ launcher
[Launcher started]
Available apps: 0=Terminal, 1=Painter
Enter app number: 0
[Starting /bin/terminal]
```

✅ **应用启动器工作正常**！

---

## ⚙️ 八、扩展方向

1. **鼠标事件集成**  
   - 通过 Display Server 接收鼠标点击坐标
   - 判断点击位置是否在图标区域内

2. **图标缓存**  
   - 首次加载后缓存到内存，避免重复读取

3. **应用分类与搜索**  
   - 按 Categories 分组显示
   - 支持关键字搜索

4. **动态应用安装**  
   - 监控 `/usr/share/applications/` 目录变化
   - 自动更新 Launcher 界面

> 💡 **GNOME Launcher、Windows Start Menu 均基于此思想**！

---

## 💬 写在最后

Launcher 是桌面环境的**门户**。  
它将分散的应用整合为直观的入口，  
让用户无需记忆路径即可启动程序。

今天你实现的第一个应用按钮，  
正是无数桌面环境的起点。

> 🌟 **好的设计，是让用户感觉不到设计的存在。**

---

📬 **动手挑战**：  
添加第三个应用（Calculator），并为其创建 .desktop 文件和图标。  
欢迎在评论区分享你的 Launcher 界面截图！

👇 下一篇你想看：**鼠标事件处理与窗口拖拽**，还是 **字体渲染与文本输入**？

---

**#操作系统 #内核开发 #Launcher #桌面环境 #.desktop #应用管理 #图形界面 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“launcher”**，获取：
> - 完整 Launcher 源码（含 .desktop 解析器）
> - 示例 .desktop 文件模板
> - PNG 图标加载简化实现
