## 高通平台

```bash
adb reboot bootloader
fastboot boot kernel.bin ramdisk.img
```

for kernel.bin:

```cpp
void main() {
    // Assume the bootloader has already initialized the framebuffer
    volatile unsigned int *framebuffer = (unsigned int *)0x80000000; // Example framebuffer address
    char message[] = "Hello, World!";
    for (int i = 0; message[i] != '\0'; i++) {
        framebuffer[i] = 0x00FFFFFF; // White color (RGB format)
        framebuffer[i + 1] = message[i]; // ASCII character
    }

    while (1); // Infinite loop to prevent reboot
}
```

compilation:

```bash
aarch64-linux-gnu-gcc -ffreestanding -nostdlib -o my_kernel my_kernel.c
aarch64-linux-gnu-objcopy -O binary my_kernel my_kernel.bin
```

boot:

```bash
fastboot pull boot boot.img
unpackbootimg -i boot.img -o extracted/
fastboot boot my_kernel.bin extracted/ramdisk.cpio.gz
```

framebuffer address:

```bash
adb shell cat /proc/device-tree/firmware/qcom,mdss-fb
```

如果在你的设备上执行 `adb shell cat /proc/device-tree/firmware/qcom,mdss-fb` 没有找到对应的节点，这可能是因为：

1. **设备树节点路径不同**：不同的设备和 SoC 可能有不同的设备树结构。你可能需要查找其他相关的节点。
2. **设备树未启用或不可访问**：某些设备可能不公开设备树信息，或者设备树信息被隐藏。
3. **显示控制器名称不同**：Qualcomm 平台的显示控制器可能使用不同的命名约定。

---

### **1. 如何找到正确的 Framebuffer 信息？**

#### **方法 1: 检查 `/dev/graphics/fb*`**
Android 设备通常会将 framebuffer 映射到 `/dev/graphics/fb0` 或类似的路径。你可以通过以下命令检查是否存在 framebuffer 设备：
```bash
adb shell ls -l /dev/graphics/
```
输出示例：
```
crw-rw---- 1 root graphics 29, 0 Jan 1 00:00 fb0
```
如果存在 `fb0`，你可以尝试读取它的内容：
```bash
adb shell dd if=/dev/graphics/fb0 of=/sdcard/framebuffer.raw bs=1 count=1024
adb pull /sdcard/framebuffer.raw
```
然后使用工具（如 `ffplay` 或 `ImageMagick`）分析 raw 文件的内容。

---

#### **方法 2: 查找其他设备树节点**
设备树中的 framebuffer 节点可能位于不同的路径。你可以尝试搜索整个设备树：
```bash
adb shell find /proc/device-tree -type f | grep -i fb
```
或者直接列出所有节点：
```bash
adb shell find /proc/device-tree -type f
```
寻找与显示相关的关键词，例如 `mdss`, `mdp`, `dsi`, `fb`, 或者 `display`。

---

#### **方法 3: 使用 `dmesg` 查看 Kernel Log**
Kernel 启动时通常会打印有关 framebuffer 的初始化信息。你可以通过以下命令查看日志：
```bash
adb shell dmesg | grep -i fb
```
输出示例：
```
[    1.234567] msm_fb: framebuffer registered at 0x80000000
```
这些日志可能会提供 framebuffer 的基地址或其他相关信息。

---

#### **方法 4: 检查 `/sys/class/graphics/`**
在某些设备上，framebuffer 的信息可以通过 `/sys/class/graphics/` 获取：
```bash
adb shell ls -l /sys/class/graphics/
```
输出示例：
```
lrwxrwxrwx 1 root root 0 Jan 1 00:00 fb0 -> ../../devices/platform/soc/xxx/fb0
```
你可以进一步检查 `fb0` 的属性：
```bash
adb shell cat /sys/class/graphics/fb0/virtual_size
adb shell cat /sys/class/graphics/fb0/stride
```

---

### **2. 如果无法找到 Framebuffer 地址怎么办？**

如果上述方法都无法找到 framebuffer 的具体地址，你可以尝试以下替代方案：

#### **(1) 假设默认地址**
一些 Qualcomm 平台的 framebuffer 默认地址是固定的。例如：
- `0x80000000` 是一个常见的起始地址。
- 尝试从这个地址开始写入数据，看看是否能在屏幕上显示内容。

#### **(2) 使用 Bootloader 提供的 Framebuffer**
许多 Qualcomm bootloader（如 `lk`）会在启动内核之前初始化显示硬件，并将 framebuffer 地址传递给内核。你可以通过以下方式获取该地址：
- 检查 kernel 命令行参数（`cmdline`）中是否包含 framebuffer 相关信息：
  ```bash
  adb shell cat /proc/cmdline
  ```
  输出示例：
  ```
  console=ttyMSM0,115200n8 androidboot.hardware=qcom msm_rtb.filter=0x237 ...
  ```
  如果其中包含类似 `framebuffer=0x80000000` 的参数，则说明 bootloader 已经初始化了 framebuffer。

#### **(3) 逆向工程 Bootloader**
如果你有 bootloader 的源码或二进制文件，可以分析它如何初始化显示硬件，并提取 framebuffer 地址。

---

### **3. 编写代码测试 Framebuffer**

假设你找到了 framebuffer 地址（例如 `0x80000000`），可以编写一个简单的程序来测试显示内容。

#### **示例代码**
```c
#include <stdint.h>
#include <string.h>

#define FRAMEBUFFER_ADDRESS 0x80000000
#define SCREEN_WIDTH 1080
#define SCREEN_HEIGHT 1920
#define PIXEL_SIZE 4 // ARGB format (4 bytes per pixel)

void draw_pixel(volatile uint32_t *framebuffer, int x, int y, uint32_t color) {
    framebuffer[y * SCREEN_WIDTH + x] = color;
}

int main() {
    volatile uint32_t *framebuffer = (uint32_t *)FRAMEBUFFER_ADDRESS;

    // 清屏为黑色
    memset((void *)framebuffer, 0, SCREEN_WIDTH * SCREEN_HEIGHT * PIXEL_SIZE);

    // 在屏幕中央绘制红色像素
    int x = SCREEN_WIDTH / 2;
    int y = SCREEN_HEIGHT / 2;
    draw_pixel(framebuffer, x, y, 0xFFFF0000); // 红色 (ARGB)

    while (1); // 防止程序退出
}
```

#### **编译并运行**
1. **交叉编译**：
   ```bash
   aarch64-linux-gnu-gcc -ffreestanding -nostdlib -o test_framebuffer test_framebuffer.c
   aarch64-linux-gnu-objcopy -O binary test_framebuffer test_framebuffer.bin
   ```

2. **测试**：
   使用 `fastboot boot` 测试：
   ```bash
   fastboot boot test_framebuffer.bin ramdisk.img
   ```

---

### **4. 如果仍然无法显示内容怎么办？**

如果上述方法都无法让你显示内容，可能需要更深入地了解你的设备硬件和 bootloader 的行为。以下是一些建议：

1. **查阅文档**：
   查阅你的设备的 SoC 文档（如 Qualcomm 的技术参考手册），了解显示控制器的工作原理。

2. **调试硬件**：
   使用 JTAG 或 UART 进行硬件级调试，观察 bootloader 和 kernel 的行为。

3. **社区支持**：
   访问 XDA Developers 或其他开发者论坛，寻找与你的设备相关的信息。

---

### **总结**

1. **优先查找设备树或系统文件**：
   使用 `/proc/device-tree`, `/sys/class/graphics/`, 或 `dmesg` 找到 framebuffer 的地址。

2. **假设默认地址**：
   如果找不到明确的地址，可以尝试常见的默认地址（如 `0x80000000`）。

3. **利用 Bootloader 的初始化**：
   如果 bootloader 已经初始化了显示硬件，尝试从 kernel 命令行或内存映射中获取 framebuffer 地址。

4. **测试和验证**：
   编写简单的代码测试 framebuffer 功能，确保能够正确显示内容。

如果你能提供更多关于设备的具体信息（如设备型号、SoC 类型等），我可以为你提供更有针对性的建议！