## 基础篇 - 盲目的玩具

- [001-linker-and-assmebly.md](Notes/009-kernel-from-scratch/001-linker-and-assmebly.md) — 链接器与汇编基础  
- [002-helloworld-explained.md](Notes/009-kernel-from-scratch/002-helloworld-explained.md) — 最小内核 Hello World 详解  
- [003-multiboot-in-detail.md](Notes/009-kernel-from-scratch/003-multiboot-in-detail.md) — Multiboot 规范详解  
- [004-protected-model-segmented-memory-management.md](Notes/009-kernel-from-scratch/004-protected-model-segmented-memory-management.md) — 保护模式与分段内存管理  
- [005-paging-mode.md](Notes/009-kernel-from-scratch/005-paging-mode.md) — 分页模式与页表  
- [006-task-switch.md](Notes/009-kernel-from-scratch/006-task-switch.md) — 任务切换机制  
- [007-user-mode.md](Notes/009-kernel-from-scratch/007-user-mode.md) — 用户态与权限分级  
- [008-fork-exec.md](Notes/009-kernel-from-scratch/008-fork-exec.md) — 进程创建与执行（fork/exec）  
- [009-simple-file-system.md](Notes/009-kernel-from-scratch/009-simple-file-system.md) — 简易文件系统实现  
- [010-vfs-and-ext2.md](Notes/009-kernel-from-scratch/010-vfs-and-ext2.md) — VFS 框架与 Ext2  
- [011-block-device.md](Notes/009-kernel-from-scratch/011-block-device.md) — 块设备与驱动接口  
- [012-stdout-stdin.md](Notes/009-kernel-from-scratch/012-stdout-stdin.md) — 标准输入输出（stdin/stdout）  
- [013-sysfs.md](Notes/009-kernel-from-scratch/013-sysfs.md) — sysfs 虚拟文件系统  
- [014-memory-management-1.md](Notes/009-kernel-from-scratch/014-memory-management-1.md) — 内存管理（一）  
- [015-slab.md](Notes/009-kernel-from-scratch/015-slab.md) — SLAB 分配器  
- [016-vma-process-release.md](Notes/009-kernel-from-scratch/016-vma-process-release.md) — VMA 与进程释放  
- [017-scheduling.md](Notes/009-kernel-from-scratch/017-scheduling.md) — 调度器基础  
- [018-smp.md](Notes/009-kernel-from-scratch/018-smp.md) — 对称多处理（SMP）  
- [019-address-space.md](Notes/009-kernel-from-scratch/019-address-space.md) — 进程地址空间  
- [020-utils.md](Notes/009-kernel-from-scratch/020-utils.md) — 工具与辅助模块  
- [021-terminal.md](Notes/009-kernel-from-scratch/021-terminal.md) — 终端子系统  
- [022-device-driver-modeling.md](Notes/009-kernel-from-scratch/022-device-driver-modeling.md) — 设备驱动建模  
- [023-framebuffer.md](Notes/009-kernel-from-scratch/023-framebuffer.md) — 帧缓冲驱动  
- [024-display-server-shared-memory.md](Notes/009-kernel-from-scratch/024-display-server-shared-memory.md) — 显示服务器与共享内存  
- [025-domain-socket.md](Notes/009-kernel-from-scratch/025-domain-socket.md) — 域套接字  
- [026-user-mode-display-server.md](Notes/009-kernel-from-scratch/026-user-mode-display-server.md) — 用户态显示服务器  
- [027-gui-app.md](Notes/009-kernel-from-scratch/027-gui-app.md) — GUI 应用实现  
- [028-window-manager.md](Notes/009-kernel-from-scratch/028-window-manager.md) — 窗口管理器  
- [029-mouse-device.md](Notes/009-kernel-from-scratch/029-mouse-device.md) — 鼠标设备  
- [030-mouse-for-window-manager.md](Notes/009-kernel-from-scratch/030-mouse-for-window-manager.md) — 窗口管理器的鼠标支持  
- [031-text-rendering.md](Notes/009-kernel-from-scratch/031-text-rendering.md) — 文本渲染  
- [032-lvgl.md](Notes/009-kernel-from-scratch/032-lvgl.md) — LVGL 移植与使用  
- [033-networking.md](Notes/009-kernel-from-scratch/033-networking.md) — 网络基础  
- [034-pcie.md](Notes/009-kernel-from-scratch/034-pcie.md) — PCIe 总线与设备  
- [035-network-stack.md](Notes/009-kernel-from-scratch/035-network-stack.md) — 网络协议栈实现  

## 扩展篇 - 完善子系统

### 内存子系统
- [100-memory-general.md](Notes/009-kernel-from-scratch/100-memory-general.md) — 内存通论  
- [101-bootup-stage-memory.md](Notes/009-kernel-from-scratch/101-bootup-stage-memory.md) — 启动阶段的内存管理  
- [102-linux-buddy.md](Notes/009-kernel-from-scratch/102-linux-buddy.md) — Linux Buddy 系统  
- [103-slab.md](Notes/009-kernel-from-scratch/103-slab.md) — SLAB 机制深入  
- [104-vma.md](Notes/009-kernel-from-scratch/104-vma.md) — VMA 机制深入  
- [105-vma-malloc.md](Notes/009-kernel-from-scratch/105-vma-malloc.md) — 基于 VMA 的内存分配  
- [106-cow-numa.md](Notes/009-kernel-from-scratch/106-cow-numa.md) — 写时复制与 NUMA  

### 文件系统
- [201-vfs.md](Notes/009-kernel-from-scratch/201-vfs.md) — VFS 设计与实现  
- [202-block-device.md](Notes/009-kernel-from-scratch/202-block-device.md) — 块设备层设计  
- [203-ext2-filesystem.md](Notes/009-kernel-from-scratch/203-ext2-filesystem.md) — Ext2 文件系统实现  
- [204-ext4.md](Notes/009-kernel-from-scratch/204-ext4.md) — Ext4 结构与特性  
- [205-fs-mount.md](Notes/009-kernel-from-scratch/205-fs-mount.md) — 文件系统挂载流程  
- [206-page-cache.md](Notes/009-kernel-from-scratch/206-page-cache.md) — 页面缓存（Page Cache）  
- [207-cow-compression-encryption.md](Notes/009-kernel-from-scratch/207-cow-compression-encryption.md) — COW/压缩/加密  

### 调度专题
- [300-scheduler.md](Notes/009-kernel-from-scratch/300-scheduler.md) — 调度器总览  
- [301-sched-basics.md](Notes/009-kernel-from-scratch/301-sched-basics.md) — 调度基础  
- [302-sched-algos.md](Notes/009-kernel-from-scratch/302-sched-algos.md) — 调度算法  
- [303-smp-load-balance.md](Notes/009-kernel-from-scratch/303-smp-load-balance.md) — SMP 负载均衡  
- [304-realtime-scheduling.md](Notes/009-kernel-from-scratch/304-realtime-scheduling.md) — 实时调度  
- [305-cfs-scheduler.md](Notes/009-kernel-from-scratch/305-cfs-scheduler.md) — CFS 调度器  
- [306-linux-sched-ext.md](Notes/009-kernel-from-scratch/306-linux-sched-ext.md) — Linux sched_ext 扩展  
- [307-node-numa-container.md](Notes/009-kernel-from-scratch/307-node-numa-container.md) — NUMA/节点/容器化  


### 权限管理

传统权限
selinux
seccomp
apparmor

## 升华篇 - 理论再实践

虚拟概念空间
资源分配的基本单位与调度的基本单位



## 附加篇 - 有趣的知识

- [900-BIOS.md](Notes/009-kernel-from-scratch/900-BIOS.md) — BIOS 与传统启动流程  


## 仓库介绍

从0写OS内核代码仓库：[WandOS: An OS kernel developed using C++](https://github.com/zhangfuwen/wandos)
欢迎加星 + 贡献代码~