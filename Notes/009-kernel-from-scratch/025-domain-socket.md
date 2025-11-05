
# 从零写 OS 内核-第二十五篇：Unix 域套接字 —— 实现进程间通信（IPC）的基石

> **“Display Server 与 Client 需要高效通信，但管道只能单向、无结构。  
> 今天，我们实现 Unix 域套接字（AF_UNIX），让进程间通信像操作文件一样自然！”**

在上一篇中，我们构建了 Display Server + Client 架构，  
但 IPC 通道是简化的管道（pipe），存在明显局限：  
- **仅支持单向通信**  
- **无连接概念** → 无法区分多个 Client  
- **无文件描述符传递** → 无法共享内存句柄  

真正的 IPC 需要 **Unix 域套接字（Unix Domain Socket）**：  
✅ **双向通信**  
✅ **支持 SOCK_STREAM（流式）和 SOCK_DGRAM（报文）**  
✅ **可传递文件描述符**（如共享内存、设备句柄）  
✅ **通过路径名寻址**（如 `/tmp/display.sock`）  

今天，我们就来：  
✅ **设计套接字 VFS 抽象层**  
✅ **实现 AF_UNIX 协议族**  
✅ **支持 connect/listen/accept/send/recv**  
✅ **演示文件描述符传递**  

让你的 OS 拥有**工业级 IPC 能力**！

---

## 🧩 一、为什么需要 Unix 域套接字？

### 管道（Pipe）的局限：
| 问题 | 后果 |
|------|------|
| **单向通信** | 需要两个管道实现双向 |
| **无连接标识** | Server 无法区分多个 Client |
| **无路径名** | 只能父子进程通信 |
| **无 FD 传递** | 无法共享资源句柄 |

### Unix 域套接字的优势：
- **路径名寻址**：`/var/run/display.sock`  
- **全双工**：单个 socket 支持双向  
- **连接导向**：`accept()` 返回新 socket 代表一个连接  
- **FD 传递**：通过 `sendmsg` 传递文件描述符  

> 💡 **D-Bus、Wayland、X11 全部基于 Unix 域套接字**！

---

## 🏗️ 二、套接字 VFS 抽象层

套接字在 VFS 中表现为**特殊文件**，但操作通过 `socket` 系统调用。

### 1. **套接字 inode**
```c
struct socket_inode {
    struct socket *sock;    // 指向套接字对象
};

struct inode *sock_alloc_inode() {
    struct inode *inode = alloc_inode();
    inode->i_mode = S_IFSOCK | 0666;
    inode->i_private = kmalloc(sizeof(struct socket_inode));
    return inode;
}
```

### 2. **套接字文件操作**
```c
static const struct file_operations socket_fops = {
    .read = sock_read,
    .write = sock_write,
    .poll = sock_poll,
    .ioctl = sock_ioctl,
    .mmap = sock_mmap,      // 用于共享内存传递
};
```

### 3. **系统调用对接**
```c
int sys_socket(int domain, int type, int protocol) {
    if (domain != AF_UNIX) return -1;
    
    struct socket *sock = sock_create(AF_UNIX, type, protocol);
    if (!sock) return -1;
    
    // 创建 socket inode
    struct inode *inode = sock_alloc_inode();
    ((struct socket_inode*)inode->i_private)->sock = sock;
    
    // 分配文件描述符
    struct file *file = alloc_file();
    file->f_inode = inode;
    file->f_op = &socket_fops;
    file->private_data = sock;
    
    return alloc_fd(file);
}
```

> 🔑 **`socket()` 返回的 fd 本质是一个特殊文件**！

---

## 🔌 三、AF_UNIX 协议实现

### 1. **套接字结构**
```c
#define UNIX_PATH_MAX 108

struct unix_sock {
    struct socket socket;
    struct unix_sock *peer;         // 对端套接字
    struct list_head listen_queue;  // 监听队列（server）
    char name[UNIX_PATH_MAX];       // 路径名
    struct inode *dentry;           // 对应的 VFS dentry
};

struct socket {
    int state;                      // SS_UNCONNECTED, SS_CONNECTED
    int type;                       // SOCK_STREAM, SOCK_DGRAM
    struct unix_sock *sk;
    spinlock_t lock;
};
```

### 2. **bind：绑定路径名**
```c
int sys_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    struct file *file = get_file(sockfd);
    struct unix_sock *u = ((struct unix_sock*)file->private_data->sk);
    
    // 1. 解析路径（addr->sun_path）
    const char *path = ((struct sockaddr_un*)addr)->sun_path;
    
    // 2. 在 VFS 中创建 socket 文件
    struct dentry *dentry = vfs_create(vfs_root, path, S_IFSOCK);
    if (!dentry) return -1;
    
    // 3. 保存路径和 dentry
    strcpy(u->name, path);
    u->dentry = dentry->inode;
    
    return 0;
}
```

### 3. **listen：监听连接**
```c
int sys_listen(int sockfd, int backlog) {
    struct file *file = get_file(sockfd);
    struct unix_sock *u = ((struct unix_sock*)file->private_data->sk);
    
    u->socket.state = SS_LISTENING;
    INIT_LIST_HEAD(&u->listen_queue);
    return 0;
}
```

### 4. **accept：接受连接**
```c
int sys_accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen) {
    struct file *file = get_file(sockfd);
    struct unix_sock *server = ((struct unix_sock*)file->private_data->sk);
    
    // 1. 从监听队列取连接
    if (list_empty(&server->listen_queue)) {
        return -1; // 或阻塞
    }
    struct unix_sock *client = list_first_entry(&server->listen_queue, struct unix_sock, link);
    list_del(&client->link);
    
    // 2. 创建新 socket 文件描述符
    struct inode *inode = sock_alloc_inode();
    ((struct socket_inode*)inode->i_private)->sock = &client->socket;
    
    struct file *new_file = alloc_file();
    new_file->f_inode = inode;
    new_file->f_op = &socket_fops;
    new_file->private_data = &client->socket;
    
    return alloc_fd(new_file);
}
```

### 5. **connect：连接 Server**
```c
int sys_connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    struct file *file = get_file(sockfd);
    struct unix_sock *client = ((struct unix_sock*)file->private_data->sk);
    
    // 1. 查找 Server socket
    const char *path = ((struct sockaddr_un*)addr)->sun_path;
    struct inode *server_inode = vfs_lookup(path);
    if (!server_inode || !S_ISSOCK(server_inode->i_mode)) {
        return -1;
    }
    
    struct unix_sock *server = ((struct socket_inode*)server_inode->i_private)->sock;
    
    // 2. 建立连接
    client->peer = server;
    server->peer = client; // 简化：实际需队列
    
    // 3. 将 client 加入 server 监听队列
    list_add_tail(&client->link, &server->listen_queue);
    
    return 0;
}
```

---

## 💬 四、数据传输：send/recv

### 1. **内核消息队列**
```c
struct sk_buff {
    struct list_head list;
    char *data;
    size_t len;
    struct file *files[MAX_SCM_FILES]; // 传递的文件描述符
    int num_files;
};

struct unix_sock {
    // ... 其他字段
    struct list_head receive_queue; // 接收队列
    struct list_head send_queue;    // 发送队列（流式）
};
```

### 2. **sendmsg（支持 FD 传递）**
```c
ssize_t sys_sendmsg(int sockfd, const struct msghdr *msg, int flags) {
    struct file *file = get_file(sockfd);
    struct unix_sock *u = ((struct unix_sock*)file->private_data->sk);
    
    // 1. 分配 skb
    struct sk_buff *skb = alloc_skb(msg->msg_iov[0].iov_len);
    memcpy(skb->data, msg->msg_iov[0].iov_base, msg->msg_iov[0].iov_len);
    
    // 2. 处理控制消息（SCM_RIGHTS = FD 传递）
    if (msg->msg_control) {
        struct cmsghdr *cmsg = CMSG_FIRSTHDR(msg);
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
            int *fds = (int*)CMSG_DATA(cmsg);
            int num_fds = (cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int);
            
            for (int i = 0; i < num_fds && i < MAX_SCM_FILES; i++) {
                // 将 fd 转换为 file 对象
                skb->files[i] = get_file(fds[i]);
                skb->num_files++;
            }
        }
    }
    
    // 3. 加入对端接收队列
    spinlock_acquire(&u->peer->lock);
    list_add_tail(&skb->list, &u->peer->receive_queue);
    spinlock_release(&u->peer->lock);
    
    return msg->msg_iov[0].iov_len;
}
```

### 3. **recvmsg（接收 FD）**
```c
ssize_t sys_recvmsg(int sockfd, struct msghdr *msg, int flags) {
    struct file *file = get_file(sockfd);
    struct unix_sock *u = ((struct unix_sock*)file->private_data->sk);
    
    // 1. 从接收队列取 skb
    if (list_empty(&u->receive_queue)) return -1;
    struct sk_buff *skb = list_first_entry(&u->receive_queue, struct sk_buff, list);
    list_del(&skb->list);
    
    // 2. 复制数据
    memcpy(msg->msg_iov[0].iov_base, skb->data, skb->len);
    
    // 3. 处理 FD 传递
    if (skb->num_files && msg->msg_control) {
        struct cmsghdr *cmsg = (struct cmsghdr*)msg->msg_control;
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(skb->num_files * sizeof(int));
        
        int *fds = (int*)CMSG_DATA(cmsg);
        for (int i = 0; i < skb->num_files; i++) {
            // 将 file 对象转换为新 fd
            fds[i] = alloc_fd(dup_file(skb->files[i]));
        }
    }
    
    free_skb(skb);
    return skb->len;
}
```

> 🔑 **FD 传递的本质：将 file 对象从一个进程的 fd_table 复制到另一个进程**！

---

## 🧪 五、测试：Display Server 与 Client 通信

### 1. **Server 创建 socket**
```c
// kernel/display_server.c
int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
struct sockaddr_un addr = {.sun_family = AF_UNIX, .sun_path = "/tmp/display.sock"};
bind(server_fd, (struct sockaddr*)&addr, sizeof(addr));
listen(server_fd, 5);
```

### 2. **Client 连接并传递 shm**
```c
// user/client.c
int client_fd = socket(AF_UNIX, SOCK_STREAM, 0);
connect(client_fd, (struct sockaddr*)&addr, sizeof(addr));

// 创建共享内存
int shm_fd = shmget(123, 4096, 0666);
void *shm = shmat(shm_fd, NULL, 0);

// 通过 socket 传递 shm_fd
char buf[1] = {0};
struct msghdr msg = {0};
struct iovec iov = {.iov_base = buf, .iov_len = 1};
msg.msg_iov = &iov;
msg.msg_iovlen = 1;

// 控制消息：传递 fd
char ctrl[CMSG_SPACE(sizeof(int))];
msg.msg_control = ctrl;
msg.msg_controllen = sizeof(ctrl);

struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
cmsg->cmsg_level = SOL_SOCKET;
cmsg->cmsg_type = SCM_RIGHTS;
cmsg->cmsg_len = CMSG_LEN(sizeof(int));
memcpy(CMSG_DATA(cmsg), &shm_fd, sizeof(int));

sendmsg(client_fd, &msg, 0);
```

### 3. **Server 接收 shm_fd**
```c
// kernel/display_server.c
char buf[1];
struct msghdr msg = {0};
// ... 初始化 msg
recvmsg(server_fd, &msg, 0);

// 从控制消息获取 fd
struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
int received_shm_fd = *(int*)CMSG_DATA(cmsg);
// 现在 Server 可访问 Client 的共享内存！
```

✅ **Client 通过 socket 安全传递共享内存句柄**！

---

## ⚠️ 六、关键实现细节

1. **引用计数**  
   - `file` 对象必须有引用计数，FD 传递时增加
   - 进程退出时自动关闭 FD

2. **同步保护**  
   - 接收/发送队列需自旋锁保护
   - 避免多核并发访问冲突

3. **路径名管理**  
   - `bind` 时创建 VFS socket 文件
   - `close` 时删除（或引用计数为 0 时删除）

4. **连接状态机**  
   - `SS_UNCONNECTED` → `SS_CONNECTING` → `SS_CONNECTED`

> 💡 **Linux 的 `af_unix.c` 是更完整的实现参考**！

---

## 💬 写在最后

Unix 域套接字是 IPC 的**瑞士军刀**。  
它统一了文件与网络接口，  
让进程通信变得简单而强大。

今天你传递的第一个文件描述符，  
正是 D-Bus、Wayland 等现代 IPC 机制的基石。

> 🌟 **最好的抽象，是让用户忘记通信的存在。**

---

📬 **动手挑战**：  
实现 `SOCK_DGRAM` 模式，并测试无连接通信。  
欢迎在评论区分享你的 IPC 性能对比！

👇 下一篇你想看：**输入事件处理与窗口管理**，还是 **字体渲染与文本显示**？

---

**#操作系统 #内核开发 #Unix域套接字 #IPC #进程间通信 #文件描述符传递 #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“socket”**，获取：
> - 完整 AF_UNIX 实现代码（含 bind/connect/accept）
> - sendmsg/recvmsg 与 FD 传递模板
> - Display Server/Client 通信集成示例
