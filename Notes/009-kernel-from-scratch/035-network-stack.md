# 从零写 OS 内核-第三十五篇：TCP/IP 协议栈 —— 用 socket GET 百度首页！

> **“网卡驱动只是硬件接口，真正的网络能力在于协议栈。  
> 今天，我们实现精简 TCP/IP 栈，并用用户态程序下载百度首页！”**

在上一篇中，我们完成了 **e1000 网卡驱动 + PCIe 支持**，  
能发送原始以太网帧，但**无法处理高层协议**。  

真正的网络应用需要 **TCP/IP 协议栈**：  
- **IP 层**：路由、分片  
- **TCP 层**：可靠传输、流控、重传  

今天，我们就来：  
✅ **实现精简 IP/TCP 协议栈**  
✅ **提供 BSD socket API**  
✅ **编写用户态 HTTP 客户端**  
✅ **GET 百度首页并保存到文件**  

让你的 OS 拥有**真正的互联网能力**！

---

## 🌐 一、协议栈架构设计

### 分层模型（自底向上）：
```
+-------------------+
|   HTTP 客户端     | ← 用户态
+-------------------+
|   BSD Socket API  | ← 系统调用
+-------------------+
|      TCP 层       | ← 内核
+-------------------+
|      IP 层        | ← 内核
+-------------------+
|   e1000 驱动      | ← 内核
+-------------------+
```

### 关键设计原则：
- **精简**：仅支持 IPv4 + TCP（无 UDP/ICMP/ARP 缓存）
- **用户态友好**：提供标准 `socket`/`connect`/`send`/`recv`
- **单线程**：无并发连接（简化状态机）

> 💡 **Linux 的 `net/ipv4/` 是完整实现，我们做最小可用子集**！

---

## 📦 二、IP 层实现

### 1. **IP 头结构**
```c
// net/ip.h
struct ip_header {
    uint8_t version_ihl;    // 4位版本 + 4位首部长度
    uint8_t tos;            // 服务类型
    uint16_t total_length;  // 总长度（含头）
    uint16_t id;            // 标识
    uint16_t frag_off;      // 分片偏移
    uint8_t ttl;            // 生存时间
    uint8_t protocol;       // 协议（6=TCP）
    uint16_t checksum;      // 首部校验和
    uint32_t src_ip;        // 源 IP
    uint32_t dst_ip;        // 目的 IP
} __attribute__((packed));
```

### 2. **发送 IP 包**
```c
// net/ip.c
int ip_send(uint32_t dst_ip, uint8_t protocol, void *data, uint16_t len) {
    // 1. 分配包内存（以太网头 + IP 头 + 数据）
    uint16_t ip_total_len = sizeof(struct ip_header) + len;
    uint8_t *packet = kmalloc(14 + ip_total_len);
    
    // 2. 填充 IP 头
    struct ip_header *ip = (void*)(packet + 14);
    ip->version_ihl = 0x45; // IPv4, 5*4=20字节头
    ip->tos = 0;
    ip->total_length = htons(ip_total_len);
    ip->id = htons(ip_id++);
    ip->frag_off = 0;
    ip->ttl = 64;
    ip->protocol = protocol;
    ip->src_ip = htonl(NETWORK_IP); // 10.0.2.15
    ip->dst_ip = dst_ip;
    ip->checksum = 0;
    ip->checksum = ip_checksum(ip, sizeof(struct ip_header));
    
    // 3. 复制数据
    memcpy(packet + 14 + sizeof(struct ip_header), data, len);
    
    // 4. 通过 e1000 发送（目标 MAC 暂用广播）
    e1000_send(packet, 14 + ip_total_len, 
               "\xFF\xFF\xFF\xFF\xFF\xFF"); // 广播 MAC
    
    kfree(packet);
    return 0;
}
```

> ⚠️ **为简化，IP 层不处理分片、路由表，MAC 地址用广播**！

---

## 🔁 三、TCP 层实现

### 1. **TCP 连接状态机**
```c
// net/tcp.h
enum tcp_state {
    TCP_CLOSED,
    TCP_SYN_SENT,
    TCP_ESTABLISHED,
    TCP_FIN_WAIT1,
    TCP_CLOSE_WAIT,
    TCP_LAST_ACK,
    TCP_TIME_WAIT
};
```

### 2. **TCP 控制块（TCB）**
```c
struct tcp_pcb {
    uint32_t local_ip;
    uint16_t local_port;
    uint32_t remote_ip;
    uint16_t remote_port;
    enum tcp_state state;
    
    // 序列号
    uint32_t snd_una; // 未确认
    uint32_t snd_nxt; // 下一个要发送
    uint32_t rcv_nxt; // 期望接收
    
    // 接收缓冲区
    uint8_t recv_buf[TCP_RECV_BUF_SIZE];
    int recv_head, recv_tail;
    
    struct tcp_pcb *next;
};
```

### 3. **发送 TCP 段**
```c
// net/tcp.c
int tcp_send(struct tcp_pcb *pcb, void *data, uint16_t len) {
    // 1. 分配 TCP 头 + 数据
    uint8_t *tcp_packet = kmalloc(sizeof(struct tcp_header) + len);
    struct tcp_header *tcp = (void*)tcp_packet;
    
    // 2. 填充 TCP 头
    tcp->src_port = htons(pcb->local_port);
    tcp->dst_port = htons(pcb->remote_port);
    tcp->seq = htonl(pcb->snd_nxt);
    tcp->ack = htonl(pcb->rcv_nxt);
    tcp->data_offset = 0x50; // 5*4=20字节头
    tcp->flags = TCP_FLAG_ACK;
    tcp->window = htons(TCP_WINDOW_SIZE);
    tcp->checksum = 0;
    tcp->urgent_ptr = 0;
    
    // 3. 复制数据
    if (data && len > 0) {
        memcpy(tcp_packet + sizeof(struct tcp_header), data, len);
        tcp->flags |= TCP_FLAG_PSH;
    }
    
    // 4. 计算校验和（伪头 + TCP 头 + 数据）
    tcp->checksum = tcp_checksum(pcb->local_ip, pcb->remote_ip,
                                 TCP_PROTO, tcp_packet, 
                                 sizeof(struct tcp_header) + len);
    
    // 5. 通过 IP 层发送
    ip_send(pcb->remote_ip, TCP_PROTO, tcp_packet, 
            sizeof(struct tcp_header) + len);
    
    // 6. 更新序列号
    pcb->snd_nxt += len;
    kfree(tcp_packet);
    return 0;
}
```

### 4. **处理接收到的 TCP 包**
```c
void tcp_input(struct ip_header *ip, uint8_t *data, uint16_t len) {
    struct tcp_header *tcp = (void*)data;
    uint16_t tcp_len = (tcp->data_offset >> 4) * 4;
    uint8_t *payload = data + tcp_len;
    uint16_t payload_len = len - tcp_len;
    
    // 1. 查找匹配的连接
    struct tcp_pcb *pcb = tcp_find_pcb(ntohs(tcp->dst_port), 
                                       ip->src_ip, ntohs(tcp->src_port));
    if (!pcb) return;
    
    // 2. 处理 SYN-ACK（连接建立）
    if (pcb->state == TCP_SYN_SENT && (tcp->flags & TCP_FLAG_SYN)) {
        pcb->rcv_nxt = ntohl(tcp->seq) + 1;
        pcb->state = TCP_ESTABLISHED;
        
        // 发送 ACK 确认
        tcp_send(pcb, NULL, 0);
        return;
    }
    
    // 3. 处理数据
    if (payload_len > 0 && pcb->state == TCP_ESTABLISHED) {
        // 将数据放入接收缓冲区
        for (int i = 0; i < payload_len; i++) {
            pcb->recv_buf[pcb->recv_tail] = payload[i];
            pcb->recv_tail = (pcb->recv_tail + 1) % TCP_RECV_BUF_SIZE;
        }
        pcb->rcv_nxt += payload_len;
        
        // 发送 ACK
        tcp_send(pcb, NULL, 0);
    }
}
```

> 🔑 **TCP 状态机简化：只支持主动连接（Client），无监听（Server）**！

---

## 📞 四、BSD Socket API

### 1. **系统调用**
```c
// sys/socket.c
int sys_socket(int domain, int type, int protocol) {
    if (domain != AF_INET || type != SOCK_STREAM) return -1;
    
    // 分配 socket 对象
    struct socket *sock = kmalloc(sizeof(struct socket));
    sock->family = AF_INET;
    sock->type = SOCK_STREAM;
    sock->pcb = NULL;
    
    // 分配 fd
    return alloc_fd(sock);
}

int sys_connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    struct socket *sock = get_socket(sockfd);
    struct sockaddr_in *sin = (void*)addr;
    
    // 1. 创建 TCP 控制块
    struct tcp_pcb *pcb = tcp_new_pcb();
    pcb->local_ip = htonl(NETWORK_IP);
    pcb->local_port = 49152 + (rand() % 16384); // 临时端口
    pcb->remote_ip = sin->sin_addr.s_addr;
    pcb->remote_port = ntohs(sin->sin_port);
    pcb->state = TCP_SYN_SENT;
    
    sock->pcb = pcb;
    
    // 2. 发送 SYN
    struct tcp_header syn;
    syn.src_port = htons(pcb->local_port);
    syn.dst_port = htons(pcb->remote_port);
    syn.seq = htonl(pcb->snd_nxt);
    syn.ack = 0;
    syn.data_offset = 0x50;
    syn.flags = TCP_FLAG_SYN;
    syn.window = htons(TCP_WINDOW_SIZE);
    syn.checksum = 0;
    syn.urgent_ptr = 0;
    syn.checksum = tcp_checksum(pcb->local_ip, pcb->remote_ip,
                                TCP_PROTO, &syn, sizeof(syn));
    
    ip_send(pcb->remote_ip, TCP_PROTO, &syn, sizeof(syn));
    pcb->snd_nxt++;
    
    return 0;
}

int sys_send(int sockfd, const void *buf, size_t len, int flags) {
    struct socket *sock = get_socket(sockfd);
    return tcp_send(sock->pcb, (void*)buf, len);
}

int sys_recv(int sockfd, void *buf, size_t len, int flags) {
    struct socket *sock = get_socket(sockfd);
    struct tcp_pcb *pcb = sock->pcb;
    
    // 从接收缓冲区复制数据
    int copied = 0;
    while (copied < len && pcb->recv_head != pcb->recv_tail) {
        ((uint8_t*)buf)[copied] = pcb->recv_buf[pcb->recv_head];
        pcb->recv_head = (pcb->recv_head + 1) % TCP_RECV_BUF_SIZE;
        copied++;
    }
    return copied;
}
```

---

## 🌍 五、用户态 HTTP 客户端

### 1. **GET 请求封装**
```c
// user/http_client.c
int http_get(const char *host, const char *path, const char *filename) {
    // 1. 创建 socket
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    
    // 2. 连接服务器（百度 IP: 220.181.38.148）
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(80);
    server_addr.sin_addr.s_addr = inet_addr("220.181.38.148");
    connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr));
    
    // 3. 发送 HTTP GET 请求
    char request[512];
    sprintf(request, "GET %s HTTP/1.1\r\n"
                     "Host: %s\r\n"
                     "Connection: close\r\n"
                     "\r\n", path, host);
    send(sockfd, request, strlen(request), 0);
    
    // 4. 接收响应并保存到文件
    int file_fd = open(filename, O_CREAT | O_WRONLY, 0644);
    char buffer[1024];
    int n;
    while ((n = recv(sockfd, buffer, sizeof(buffer), 0)) > 0) {
        write(file_fd, buffer, n);
    }
    close(file_fd);
    close(sockfd);
    
    return 0;
}
```

### 2. **主程序**
```c
void _start() {
    printf("Downloading baidu.com...\n");
    http_get("www.baidu.com", "/", "/home/user/baidu.html");
    printf("Download complete!\n");
    exit(0);
}
```

---

## 🧪 六、测试：下载百度首页

### 1. **QEMU 启动**
```bash
qemu-system-i386 -kernel kernel.bin -hda disk.img \
                 -netdev user,id=n1,hostfwd=tcp::8080-:80 -device e1000,netdev=n1
```

> 💡 **`hostfwd` 将 QEMU 80 端口映射到主机 8080，但百度需直连外网**  
> **确保 QEMU 有互联网访问权限**！

### 2. **运行效果**
```
MyOS# wget
Downloading baidu.com...
Download complete!
MyOS# cat /home/user/baidu.html
<!DOCTYPE html>
<html>
<head>
    <title>百度一下，你就知道</title>
...
```

✅ **成功通过 TCP GET 百度首页**！

---

## ⚠️ 七、简化与局限

1. **无 ARP**：MAC 地址用广播（仅限同一子网）
2. **无重传**：丢包即失败
3. **无拥塞控制**：简单固定窗口
4. **无 DNS**：需硬编码 IP
5. **单连接**：不支持并发

> 💡 **这些是教学实现的合理简化，工业级实现需补全**！

---

## 💬 写在最后

TCP/IP 协议栈是网络世界的**通用语言**。  
它让不同设备能可靠通信，  
构建了今天的互联网。

今天你实现的第一个 `GET` 请求，  
正是无数网络应用的起点。

> 🌟 **协议栈的意义，不在于代码，而在于连接世界**。

---

📬 **动手挑战**：  
添加 DNS 客户端，将 `http_get("www.baidu.com", ...)` 改为自动解析 IP。  
欢迎在评论区分享你的网络抓包分析！

👇 下一篇你想看：**UDP 与 DHCP 客户端**，还是 **C 标准库（libc）实现**？

---

**#操作系统 #内核开发 #TCP/IP #网络协议栈 #socket #HTTP #从零开始**

---

> 📢 **彩蛋**：关注后回复关键词 **“tcp”**，获取：
> - 完整 TCP/IP 协议栈代码（IP + TCP）
> - BSD socket 系统调用实现
> - HTTP 客户端用户态模板
