GRUB 寻找 Multiboot header 时，有以下规则：
### 查找位置
- **文件头部附近**：Multiboot header 必须位于内核镜像文件的前 8192（8K） 个字节内 ，且首地址需 4 字节对齐 。GRUB 会先在这个范围内查找。这是因为早期设计中，希望能快速定位到关键的启动信息，限制在小范围可提升查找效率和兼容性。比如你编写内核镜像时，就得把 Multiboot header 放在这个区间，否则 GRUB 大概率找不到。 
- **依据魔数定位**：GRUB 会查找魔数 0x1BADB002 来确定 Multiboot header 位置。找到魔数后，就从该位置开始，加载 `header_addr - load_addr` 长度的内容到 `Header_addr` 指向的物理内存，完成 header 加载。 

### 查找流程
1. GRUB 启动后，会加载内核镜像到内存。
2. 然后按上述规则，在镜像文件前 8K 字节内找魔数 0x1BADB002 。找到魔数所在偏移（设为 `header_offset`  ）后，从该偏移处开始读取 Multiboot header 相关内容。
3. 按 Multiboot 规范解析 header 信息，获取内核启动所需参数等，为后续启动内核做准备。 