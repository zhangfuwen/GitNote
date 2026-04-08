# 从零写 OS 内核-第二十六篇：用户态 Display Server 与 Client —— 真正的图形系统架构

> **“内核态 Display Server 虽可控，但违背微内核思想；真正的图形系统应在用户态！  
> 今天，我们将 Display Server 与 Client 完全移至用户态，通过 Unix 域套接字通信！”**

在上两篇中，我们实现了：  
- **Framebuffer 驱动**（内核态）  
- **Display Server + Client 架构**（Server 在内核态）  
- **Unix 域套接字**（支持 FD 传递）  

但将 Display Server 放在内核态存在严重问题：  
- **稳定性风险**：Server 崩溃 → 内核崩溃  
- **扩展性差**：无法动态更新 Server  
- **违背设计哲学**：图形系统应属用户态服务  

真正的现代图形系统（如 **Wayland**、**X11**）全部运行在**用户态**！  

今天，我们就来：  
✅ **将 Display Server 移至用户态**  
✅ **实现多 Client 并发连接**  
✅ **通过共享内存 + 套接字高效通信**  
✅ **构建 init 进程启动 Server/Client**  

让你的 OS 拥有**工业级图形架构**！

---

## 🏗️ 一、架构重构：用户态 Display Server

### 新架构图：
```
+----------------+     +----------------+     +--------------+
|  Client 1      |     |  Client 2      |     |  ...         |
| (Terminal)     |     | (Painter)      |     |              |
+-------+--------+     +-------+--------+     +--------------+
        |                      |
        | Unix Domain Socket   |
        v                      v
+-------+----------------------+--------+
|        Display Server (用户态)         |
| - 监听 /tmp/.display.sock             |
| - 管理窗口列表                        |
| - 通过 mmap 访问 Client 共享内存      |
| - 直接写 /dev/fb0 上屏                |
+----------------------------------------+
                |
                v
        +---------------+
        | /dev/fb0      | ← Framebuffer 设备（内核暴露）
        +---------------+
```

### 关键变化：
- **Display Server = 普通用户进程**  
- **Client 通过 socket 与 Server 通信**  
- **Server 通过 `/dev/fb0` 直接写屏幕**  

> 💡 **内核只提供基础服务：Framebuffer 设备 + Unix 域套接字 + 共享内存**！

---

## 🖥️ 二、用户态 Display Server 实现

### 1. **Server 主函数**
```c
// user/display_server.c
void _start() {
    // 1. 打开 Framebuffer
    int fb_fd = open("/dev/fb0", O_RDWR);
    struct fb_info fb;
    ioctl(fb_fd, FB_GET_INFO, &fb); // 获取分辨率等
    
    // 2. mmap Framebuffer 到用户空间
    void *fb_mem = mmap(NULL, fb.pitch * fb.height, 
                        PROT_READ | PROT_WRITE, MAP_SHARED, fb_fd, 0);
    close(fb_fd);
    
    // 3. 创建 Unix 域套接字
    int sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr = {
        .sun_family = AF_UNIX,
        .sun_path = "/tmp/.display.sock"
    };
    bind(sock_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(sock_fd, 10);
    
    // 4. 主循环：处理连接与事件
    while (1) {
        // 接受新 Client
        int client_fd = accept(sock_fd, NULL, NULL);
        if (client_fd >= 0) {
            // 创建新线程处理 Client
            pthread_create(&thread, NULL, handle_client, 
                          (void*)(intptr_t)client_fd);
        }
        
        // 合成并上屏（每 16ms）
        if (should_repaint()) {
            compose_and_blit(fb_mem, &fb);
        }
    }
}
```

### 2. **窗口管理（Server 端）**
```c
struct window {
    int id;
    int client_pid;
    int width, height;
    int x, y;
    void *shm_buffer;   // Client 共享内存映射
    int shm_fd;         // 共享内存 fd
    bool visible;
    struct window *next;
};

static struct window *window_list = NULL;
static int next_win_id = 1;
```

### 3. **处理 Client 命令**
```c
void* handle_client(void *arg) {
    int client_fd = (intptr_t)arg;
    
    while (1) {
        struct display_cmd cmd;
        if (recv(client_fd, &cmd, sizeof(cmd), 0) <= 0) break;
        
        switch (cmd.type) {
            case CMD_CREATE_WINDOW: {
                // 创建新窗口
                struct window *win = malloc(sizeof(struct window));
                win->id = next_win_id++;
                win->width = cmd.width;
                win->height = cmd.height;
                win->x = cmd.x; win->y = cmd.y;
                
                // 接收共享内存 fd
                char buf[1];
                struct msghdr msg = {0};
                // ... 初始化 msg
                recvmsg(client_fd, &msg, 0);
                
                struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
                int shm_fd = *(int*)CMSG_DATA(cmsg);
                
                // mmap 共享内存
                win->shm_buffer = mmap(NULL, cmd.width * cmd.height * 4,
                                       PROT_READ, MAP_SHARED, shm_fd, 0);
                win->shm_fd = shm_fd;
                win->visible = true;
                
                // 加入窗口列表
                win->next = window_list;
                window_list = win;
                
                // 返回窗口 ID
                send(client_fd, &win->id, sizeof(win->id), 0);
                break;
            }
            case CMD_COMMIT: {
                // 标记窗口需重绘
                mark_window_dirty(cmd.win_id);
                break;
            }
            case CMD_CLOSE_WINDOW: {
                close_window(cmd.win_id);
                break;
            }
        }
    }
    
    // Client 断开，清理资源
    cleanup_client_windows(client_pid);
    return NULL;
}
```

### 4. **合成与上屏**
```c
void compose_and_blit(void *fb_mem, struct fb_info *fb) {
    // 清屏
    memset(fb_mem, 0, fb->pitch * fb->height);
    
    // 遍历窗口
    for (struct window *win = window_list; win; win = win->next) {
        if (!win->visible) continue;
        
        // 从共享内存复制到 Framebuffer
        for (int y = 0; y < win->height; y++) {
            uint32_t *src = (uint32_t*)((char*)win->shm_buffer + y * win->width * 4);
            uint32_t *dst = (uint32_t*)((char*)fb_mem + (win->y + y) * fb->pitch + win->x * 4);
            memcpy(dst, src, win->width * 4);
        }
    }
}
```

> ✅ **Server 作为普通用户进程，直接操作 `/dev/fb0`**！

---

## 🖌️ 三、用户态 Client 实现

### 1. **Client 库（user/display_client.h）**
```c
typedef struct {
    int sock_fd;        // 与 Server 的 socket
    int win_id;         // 当前窗口 ID
    void *buffer;       // 共享内存 buffer
    int width, height;
} display_t;

display_t* display_connect();
int display_create_window(display_t *disp, int width, int height);
void* display_get_buffer(display_t *disp);
void display_commit(display_t *disp);
void display_close(display_t *disp);
```

### 2. **连接 Server**
```c
display_t* display_connect() {
    display_t *disp = malloc(sizeof(display_t));
    
    // 连接 Server
    disp->sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr = {
        .sun_family = AF_UNIX,
        .sun_path = "/tmp/.display.sock"
    };
    connect(disp->sock_fd, (struct sockaddr*)&addr, sizeof(addr));
    
    return disp;
}
```

### 3. **创建窗口并传递 shm**
```c
int display_create_window(display_t *disp, int width, int height) {
    // 1. 创建共享内存
    int shm_fd = shmget(IPC_PRIVATE, width * height * 4, 0666);
    void *buffer = shmat(shm_fd, NULL, 0);
    
    // 2. 发送创建命令
    struct display_cmd cmd = {
        .type = CMD_CREATE_WINDOW,
        .width = width,
        .height = height
    };
    send(disp->sock_fd, &cmd, sizeof(cmd), 0);
    
    // 3. 通过 socket 传递 shm_fd
    char buf[1] = {0};
    struct msghdr msg = {0};
    struct iovec iov = {.iov_base = buf, .iov_len = 1};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    
    char ctrl[CMSG_SPACE(sizeof(int))];
    msg.msg_control = ctrl;
    msg.msg_controllen = sizeof(ctrl);
    
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &shm_fd, sizeof(int));
    
    sendmsg(disp->sock_fd, &msg, 0);
    
    // 4. 接收窗口 ID
    int win_id;
    recv(disp->sock_fd, &win_id, sizeof(win_id), 0);
    
    disp->win_id = win_id;
    disp->buffer = buffer;
    disp->width = width;
    disp->height = height;
    
    return win_id;
}
```

### 4. **Client 绘图示例**
```c
// user/painter.c
void _start() {
    display_t *disp = display_connect();
    display_create_window(disp, 320, 240);
    
    // 绘制渐变
    uint32_t *buf = (uint32_t*)disp->buffer;
    for (int y = 0; y < 240; y++) {
        for (int x = 0; x < 320; x++) {
            buf[y * 320 + x] = (y << 16) | (x << 8) | 0xFF; // ARGB
        }
    }
    
    display_commit(disp);
    
    sleep(10);
    display_close(disp);
    exit(0);
}
```

---

## 🚀 四、系统启动：init 进程协调

### 1. **init 进程（PID=1）**
```c
// user/init.c
void _start() {
    // 1. 挂载 devfs
    mount("devfs", "/dev", "");
    
    // 2. 启动 Display Server
    if (fork() == 0) {
        char *args[] = {"/bin/display_server", NULL};
        exec("/bin/display_server", args);
    }
    
    // 3. 等待 Server 启动
    sleep(1);
    
    // 4. 启动 Terminal Client
    if (fork() == 0) {
        char *args[] = {"/bin/terminal", NULL};
        exec("/bin/terminal", args);
    }
    
    // 5. 启动 Painter Client
    if (fork() == 0) {
        char *args[] = {"/bin/painter", NULL};
        exec("/bin/painter", args);
    }
    
    // 6. 主 shell
    char *shell_args[] = {"/bin/shell", NULL};
    exec("/bin/shell", shell_args);
}
```

### 2. **文件系统布局**
```
/bin/
  ├── display_server
  ├── terminal
  ├── painter
  └── shell
/tmp/          # 用于 socket 文件
```

---

## 🧪 五、运行效果

1. **系统启动** → `init` 启动 `display_server`  
2. **Server 创建 `/tmp/.display.sock`**  
3. **Terminal/Painter 启动并连接 Server**  
4. **Client 创建窗口，传递共享内存**  
5. **Server 合成多个窗口到 `/dev/fb0`**  

屏幕显示：
- 左上：Terminal（文本界面）
- 右下：Painter（彩色渐变矩形）

✅ **完全用户态图形系统运行成功**！

---

## ⚠️ 六、优势与挑战

### 优势：
- **稳定性**：Server 崩溃 ≠ 内核崩溃
- **灵活性**：可动态替换 Server（如从 Wayland 切到 X11）
- **安全性**：Server 无特权，受用户权限限制
- **开发友好**：用 C/C++ 开发，无需内核模块

### 挑战：
- **性能**：用户态上下文切换开销
- **权限**：需确保 Server 有 `/dev/fb0` 写权限
- **同步**：多 Client 并发需线程安全

> 💡 **现代 Linux 桌面（Wayland）正是此架构**！

---

## 💬 写在最后

将 Display Server 移至用户态，  
是操作系统架构的**关键跃迁**。  
它遵循了**微内核哲学**：  
> **内核只做最核心的事，其余交给用户态服务**。

今天你启动的第一个用户态 Server，  
正是 Wayland、X11、Android SurfaceFlinger 的简化版。

> 🌟 **真正的系统之美，在于分层与解耦**。

---

📬 **动手挑战**：  
添加窗口移动功能（Client 发送 `CMD_MOVE`，Server 更新 x/y）。  
欢迎在评论区分享你的多窗口交互视频！

👇 下一篇你想看：**输入事件处理（键盘/鼠标）**，还是 **字体渲染与文本显示**？

---

**#操作系统 #内核开发 #DisplayServer #用户态图形 #Unix域套接字 #共享内存 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“user_display”**，获取：
> - 完整用户态 Display Server/Client 源码
> - init 进程启动脚本
> - 多 Client 并发测试程序
