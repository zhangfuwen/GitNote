# 从零写 OS 内核-第二十三篇：Framebuffer —— 让你的 OS 跑在图形界面上！

> **“VGA 文本模式只能显示字符，真正的图形界面需要像素级控制。  
> 今天，我们实现 Framebuffer 驱动，让你的 OS 能画点、线、矩形，甚至显示图片！”**

在前面的篇章中，我们的输出仅限于 **VGA 文本模式**（0xB8000 显存）：  
- 只能显示 ASCII 字符  
- 无法绘制图形  
- 无颜色深度控制  

而现代操作系统都运行在**图形模式**下，这依赖 **Framebuffer（帧缓冲）**：  
> **一块连续的物理内存，直接映射到屏幕像素**。

今天，我们就来：  
✅ **启用 VESA 图形模式**  
✅ **实现 Framebuffer 驱动**  
✅ **提供画点、画线、清屏等基础绘图 API**  
✅ **通过 `/dev/fb0` 暴露给用户空间**  

让你的 OS 拥有**真正的图形能力**！

---

## 🖼️ 一、什么是 Framebuffer？

**Framebuffer** 是一块**物理内存区域**，其内容直接对应屏幕像素：  
- **地址**：由 BIOS 或 UEFI 提供  
- **宽度/高度**：分辨率（如 1024×768）  
- **颜色深度**：每像素字节数（如 32 位 = RGBA）  
- **像素格式**：RGB 顺序、alpha 通道等  

### Framebuffer 优势：
- **简单**：无需复杂 GPU 驱动  
- **通用**：所有 x86 PC 支持 VESA BIOS Extensions（VBE）  
- **高效**：直接写内存 = 直接显示

> 💡 **Linux 的 `/dev/fb0`、Android 的 SurfaceFlinger 都基于 Framebuffer**！

---

## 🔌 二、通过 Multiboot 获取 Framebuffer 信息

GRUB 支持 **Multiboot 2**，可传递 Framebuffer 信息给内核。

### Multiboot 2 Framebuffer Tag 结构：
```c
struct multiboot_framebuffer {
    uint32_t type;          // 1 = RGB, 2 = EGA 文本（忽略）
    uint32_t width;         // 像素宽度
    uint32_t height;        // 像素高度
    uint32_t pitch;         // 每行字节数（= width * bytes_per_pixel）
    uint32_t bpp;           // 每像素位数（如 32）
    uint8_t red_field_pos;  // 红色通道位偏移
    uint8_t red_mask_size;
    uint8_t green_field_pos;
    uint8_t green_mask_size;
    uint8_t blue_field_pos;
    uint8_t blue_mask_size;
    uint8_t reserved;
    uint8_t reserved1;
    uint32_t addr;          // Framebuffer 物理地址
};
```

### 内核获取 Framebuffer：
```c
// kernel.c
void kernel_main(uint32_t magic, uint32_t info_addr) {
    if (magic != 0x36D76289) return; // Multiboot 2 魔数
    
    struct multiboot_info *info = (void*)info_addr;
    struct multiboot_framebuffer *fb = NULL;
    
    // 遍历 tags
    for (uint32_t tag = (uint32_t)info->tags;
         tag < (uint32_t)info + info->total_size;
         tag += ((struct multiboot_tag*)tag)->size + 7) & ~7) {
        struct multiboot_tag *t = (void*)tag;
        if (t->type == MULTIBOOT_TAG_TYPE_FRAMEBUFFER) {
            fb = (void*)t;
            break;
        }
    }
    
    if (fb && fb->type == 1) {
        framebuffer_init(
            fb->addr,
            fb->width,
            fb->height,
            fb->pitch,
            fb->bpp,
            fb->red_field_pos, fb->red_mask_size,
            fb->green_field_pos, fb->green_mask_size,
            fb->blue_field_pos, fb->blue_mask_size
        );
    }
}
```

> ✅ **GRUB 启动时需指定图形模式**：  
> `grub.cfg` 中添加：  
> `set gfxmode=1024x768x32`  
> `insmod all_video`

---

## 🖌️ 三、Framebuffer 驱动实现

### 1. **驱动数据结构**
```c
struct fb_info {
    void *vaddr;            // 虚拟地址（映射后）
    uint32_t width;
    uint32_t height;
    uint32_t pitch;         // 每行字节数
    uint8_t bpp;            // 每像素字节数（bpp/8）
    
    // 颜色通道信息
    uint8_t red_pos, red_len;
    uint8_t green_pos, green_len;
    uint8_t blue_pos, blue_len;
    
    // 当前颜色（用于绘图）
    uint32_t color;
};

static struct fb_info fb;
```

### 2. **初始化**
```c
void framebuffer_init(
    uint32_t paddr, uint32_t width, uint32_t height, uint32_t pitch,
    uint8_t bpp, 
    uint8_t red_pos, uint8_t red_len,
    uint8_t green_pos, uint8_t green_len,
    uint8_t blue_pos, uint8_t blue_len
) {
    // 1. 保存参数
    fb.width = width;
    fb.height = height;
    fb.pitch = pitch;
    fb.bpp = bpp / 8;
    fb.red_pos = red_pos; fb.red_len = red_len;
    fb.green_pos = green_pos; fb.green_len = green_len;
    fb.blue_pos = blue_pos; fb.blue_len = blue_len;
    
    // 2. 映射物理地址到内核高半空间
    fb.vaddr = (void*)(paddr + KERNEL_VIRTUAL_BASE);
    map_range(paddr, (uint32_t)fb.vaddr, pitch * height, 
              PAGE_PRESENT | PAGE_RW | PAGE_GLOBAL);
    
    // 3. 设置默认颜色（白色）
    fb.color = 0xFFFFFFFF;
    
    // 4. 清屏
    fb_clear();
}
```

### 3. **基础绘图 API**
```c
// 设置像素颜色
void fb_putpixel(int x, int y, uint32_t color) {
    if (x < 0 || x >= fb.width || y < 0 || y >= fb.height) return;
    
    uint8_t *pixel = (uint8_t*)fb.vaddr + y * fb.pitch + x * fb.bpp;
    
    if (fb.bpp == 4) {
        *(uint32_t*)pixel = color;
    } else if (fb.bpp == 2) {
        *(uint16_t*)pixel = color_to_rgb565(color);
    }
}

// 获取像素颜色
uint32_t fb_getpixel(int x, int y) {
    if (x < 0 || x >= fb.width || y < 0 || y >= fb.height) return 0;
    uint8_t *pixel = (uint8_t*)fb.vaddr + y * fb.pitch + x * fb.bpp;
    return (fb.bpp == 4) ? *(uint32_t*)pixel : 0;
}

// 清屏
void fb_clear() {
    uint32_t size = fb.pitch * fb.height;
    memset(fb.vaddr, 0, size);
}

// 画线（Bresenham 算法）
void fb_line(int x0, int y0, int x1, int y1) {
    // ... 标准实现
    for (...) {
        fb_putpixel(x, y, fb.color);
    }
}

// 画矩形
void fb_rect(int x, int y, int w, int h, bool fill) {
    if (fill) {
        for (int i = y; i < y + h; i++) {
            for (int j = x; j < x + w; j++) {
                fb_putpixel(j, i, fb.color);
            }
        }
    } else {
        fb_line(x, y, x + w, y);
        fb_line(x, y, x, y + h);
        fb_line(x + w, y, x + w, y + h);
        fb_line(x, y + h, x + w, y + h);
    }
}
```

---

## 📁 四、通过 devfs 暴露 Framebuffer

### 1. **注册 Framebuffer 设备**
```c
// drivers/fbdev.c
static struct file_operations fb_fops = {
    .read = fb_read,
    .write = fb_write,
    .mmap = fb_mmap,
};

void fbdev_init() {
    struct device *fbdev = kmalloc(sizeof(struct device));
    strcpy(fbdev->name, "fb0");
    fbdev->devt = mkdev(MAJOR_FB, 0);
    fbdev->driver_data = &fb;
    
    device_register(fbdev);
    register_chrdev(MAJOR_FB, "fb", &fb_fops);
}
```

### 2. **实现 mmap（关键！）**
```c
int fb_mmap(struct file *file, struct vm_area_struct *vma) {
    struct fb_info *info = ((struct device*)file->f_inode->i_private)->driver_data;
    
    // 将 Framebuffer 物理地址映射到用户空间
    uint32_t paddr = (uint32_t)info->vaddr - KERNEL_VIRTUAL_BASE;
    uint32_t offset = vma->vm_pgoff * PAGE_SIZE;
    uint32_t size = vma->vm_end - vma->vm_start;
    
    for (uint32_t i = 0; i < size; i += PAGE_SIZE) {
        map_page(vma->vm_task->cr3, vma->vm_start + i, 
                 paddr + offset + i, 
                 PAGE_PRESENT | PAGE_RW | PAGE_USER);
    }
    
    return 0;
}
```

> 🔑 **`mmap` 让用户程序直接访问 Framebuffer 内存，实现高性能绘图**！

---

## 🧪 五、用户空间绘图程序

### 1. **用户态 Framebuffer 库**
```c
// user/fb.h
typedef struct {
    void *mem;
    int width, height, pitch, bpp;
} fb_device_t;

fb_device_t* fb_open() {
    int fd = open("/dev/fb0", O_RDWR);
    fb_device_t *fb = malloc(sizeof(fb_device_t));
    
    // 获取 Framebuffer 信息（通过 ioctl 或固定值）
    fb->width = 1024; fb->height = 768;
    fb->pitch = 1024 * 4; fb->bpp = 4;
    
    // mmap 到用户空间
    fb->mem = mmap(NULL, fb->pitch * fb->height, 
                   PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    return fb;
}

void fb_putpixel(fb_device_t *fb, int x, int y, uint32_t color) {
    if (x < 0 || x >= fb->width || y < 0 || y >= fb->height) return;
    uint32_t *pixel = (uint32_t*)((char*)fb->mem + y * fb->pitch + x * 4);
    *pixel = color;
}
```

### 2. **绘图测试程序**
```c
// user/draw.c
void _start() {
    fb_device_t *fb = fb_open();
    
    // 画渐变背景
    for (int y = 0; y < fb->height; y++) {
        uint32_t color = (y << 16) | (y << 8) | y; // RGB 渐变
        for (int x = 0; x < fb->width; x++) {
            fb_putpixel(fb, x, y, color);
        }
    }
    
    // 画矩形
    for (int i = 0; i < 100; i++) {
        fb_putpixel(fb, 100 + i, 100, 0xFF0000FF); // 蓝色
    }
    
    exit(0);
}
```

### 运行效果：
- 屏幕显示 RGB 渐变背景
- 左上角有一条蓝色横线

✅ **用户程序直接操作像素**！

---

## ⚠️ 六、高级话题

1. **双缓冲（Double Buffering）**  
   - 避免绘图闪烁
   - 用户绘图到离屏缓冲，完成后整体复制到 Framebuffer

2. **硬件加速**  
   - 利用 GPU 的 BitBLT、填充指令
   - 需要更复杂的驱动（如 DRM/KMS）

3. **多显示器支持**  
   - `/dev/fb0`, `/dev/fb1` 对应不同屏幕
   - 通过 EDID 获取显示器信息

4. **模式切换**  
   - 动态改变分辨率（需 VBE 支持）

> 💡 **Framebuffer 是图形栈的最底层，之上可构建 GUI（如 MiniGUI、LVGL）**！

---

## 💬 写在最后

Framebuffer 是通往图形世界的**第一块基石**。  
它简单、通用、高效，  
让操作系统无需复杂 GPU 驱动就能显示像素。

今天你绘制的第一个蓝色像素，  
正是无数图形界面的起点。

> 🌟 **图形界面的灵魂，始于对每一个像素的掌控。**

---

📬 **动手挑战**：  
实现 `fb_line` 函数，并画出一个三角形。  
欢迎在评论区分享你的图形作品截图！

👇 下一篇你想看：**简单 GUI 框架**，还是 **字体渲染与文本显示**？

---

**#操作系统 #内核开发 #Framebuffer #图形界面 #VESA #绘图 #devfs #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“fb”**，获取：
> - 完整 Framebuffer 驱动代码（含 Multiboot 2 解析）
> - 用户态 mmap 绘图库模板
> - GRUB 图形模式配置指南
