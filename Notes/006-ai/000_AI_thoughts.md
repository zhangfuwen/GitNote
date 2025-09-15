---
title: 000. AI 个人思考（稿纸）
---

## 推理引擎关键问题

1. 内存容量
	1. 权重
		1. 量化
		2. 蒸馏
	2. kv cache
		1. 缓存 sglang的radix tree
		2. 内存碎片化 paged attention
2. 算力
	1. prefill阶段是计算密集型的，核心是如何充分利用硬件资源
3. 内存带宽
	1. decode阶段是访存密集型的，核心是如何加载一次权重，计算更多token
	2. batching
	3. 预测性推理 [万字综述 10+ 种 LLM 投机采样推理加速方案 - 53AI-AI知识库\|大模型知识库\|大模型训练\|智能体开发](https://www.53ai.com/news/finetuning/2024071109285.html)
		1. 投机采样
		2. 多头美杜莎
		3. LookAHead  [GitHub - hao-ai-lab/LookaheadDecoding: \[ICML 2024\] Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://github.com/hao-ai-lab/LookaheadDecoding)

## 效果

### 思考更长时间

Think Twice

[Think Longer](https://www.alphaxiv.org/abs/2503.23803)

## 红杉关于AI的判断

1. 我们处于认知革命的时代，蒸汽机-> 工厂流水线，通用技术专业化改造，适应特定行业的生产需求
2. 美国有10万亿的服务业市场需要被改造
3. 投资趋势
	1. 杠杆优于确定性
	2. 真实世界验证，在真实世界证明技术能力（如Expo在HackerOne打赢真实黑客）是新的黄金标准
	3. AI进入物理世界
	4. 计算成为新的生产函数


红衫投资的重点方向：

1. 持久记忆
2. 无缝通信协议
3. AI语音
4. AI安全
5. 开源AI

## Rust：个人开发者的终极赋能和战略选择

个人开发者的生存挑战：

-  **资源绝对稀缺**：缺乏测试、QA、运维团队，一人承担所有角色。
    
- **容错率极低**：一个深夜难以调试的线上 `Segmentatiοn Fault` 可能导致项目失败。
    
- **竞争压力巨大**：必须在功能、性能或可靠性上建立独特优势，方能与大型团队产品竞争。
    
- **维护成本高昂**：「**代码债**」的利息对个人而言尤为沉重，后期修复 Bug 的时间远超开发时间。
    
- **技术栈选择焦虑**：选择不当的技术栈可能导致项目中途搁浅或推倒重来。

传统的解释型语言（如 Python、JavaScript）或托管语言（如 Go、Java）在易用性上各有优势，但往往需要开发者在这些挑战上做出妥协。**Rust 的出现，从根本上提供了另一种范式：它通过将复杂性前置到编译时，换取开发整个生命周期内的绝对稳定性和极致性能，从而成为个人开发者破局的最强战略武器。**


对个人开发者的价值：

1. **竞争力优势**：你一个人写的程序，可以在性能上媲美甚至超越一个团队精心优化的 C++ 程序。这是个人项目最直接的卖点。

2. **开发效率与运行效率的统一**：你无需在「写得快」和「跑得快」之间做选择题。你可以同时拥有开发效率和顶级性能。

3. **成本控制**：高性能意味着更低的服务器配置要求。一个用 Rust 编写的 API 网关可能单核就能处理数万请求，直接 translates to 真金白银的云服务费用节省。

#### **卓越的工具链：工业化标准的基础设施，开箱即用**

- `cargo new`：创建标准化的项目结构。
    
- `cargo build`：依赖解析和构建。
    
- `cargo run`：运行。
    
- `cargo test`：集成和单元测试（内置测试框架！）。
    
- `cargo clippy`：提供大量的代码 lint 规则，教你写出更地道、更高效的 Rust 代码。
    
- `cargo fmt`：统一的代码格式化，杜绝项目中的格式争论。
    
- `cargo doc --open`：自动生成极其精美的 API 文档网站，并支持文档示例代码测试。
    

- **Cargo**：远不止是包管理器。它是**统一的项目生命周期管理工具**。
    
- **Rustup**：无缝的工具链管理。轻松安装、更新和切换稳定版、测试版和夜间版的编译器。
    
- ** crates.io**：官方包注册中心。拥有庞大且高质量的生態系統（截至2023年超过10万个包），从 Web 框架到加密货币算法，应有尽有。#### **强大的生态系统与无所不能的适用性**

#### **强大的生态系统与无所不能的适用性**
- **Web 后端**：`Axum`、`Rocket`、`Actix-web` 等框架提供了现代化、高性能的体验，媲美任何主流语言框架。
    
- **前端/WebAssembly**：Rust 是编译为 WASM 的顶级语言。你可以用 `yew`、`leptos` 等框架开发整个前端应用，或编写高性能的 WASM 模块供 JavaScript 调用。
    
- **命令行工具 (CLI)**：Rust 的静态编译、高性能和丰富的解析库（如 `clap`）使其成为编写 CLI 工具的不二之选。成功的例子包括：`ripgrep` (比 grep 更快)、`fd` (比 find 更友好)、`bat` (带语法高亮的 cat)、`starship` (超快的终端提示符)。
    
- **桌面应用**：通过 `Tauri` 框架，可以用 Web 技术（HTML/CSS/JS）构建前端，用 Rust 构建后端核心，生成体积极小、内存占用极低、性能极高的跨平台桌面应用。
    
- **系统与嵌入式**：这是 Rust 的老本行，在操作系统、数据库、游戏引擎、区块链、嵌入式设备等领域占据主导地位。

### 问题

**编译速度**

- **挑战**：严格的编译时检查导致编译速度，尤其是增量编译和全量编译，慢于 Go 或 Swift。
- **应对**：
- 使用 `sccache` 缓存编译结果。
- 在开发时使用 `cargo check` 进行快速语法检查，而非每次都 `cargo build`。
- 工具链在持续优化，速度在不断改善。

## 企业

企业的使命、愿景是不能变的，在重大决策中发挥作用。

企业的规章、制度是为了践行使命、愿景的。需要不断的强化使命、愿景。


## 编程模式

Typescript的编译器是全用函数实现的。

应该每个类仅实现非常少的功能？
用namespace使函数的查找更容易？oboe是这样的

```cpp
namespace mp3 {
struct mp3_obj {
	int n_chans;
	int n_frames;
	
	int pos;
	float *read_frame(int size);	
};
mp3_obj init_from_file(char *path) {}



};
```


## 快捷键

### 桌面级

### 窗口级(wezterm)

```lua
{ key = 'p', mods = 'CTRL|SHIFT', action = act.ActivateCommandPalette },
{ key = '?', mods = 'CTRL|SHIFT',action = act.ShowDebugOverlay },
{ key = 'o', mods = 'CTRL|SHIFT', action = act.ShowLauncher },
{ key = 'h', mods = 'CTRL|SHIFT', action = act.ActivateTabRelative(-1) }, 
{ key = 'l', mods = 'CTRL|SHIFT', action = act.ActivateTabRelative(1)},
{ key = '{', mods = 'CTRL|SHIFT', action = act.MoveTabRelative(-1) },
{ key = '}', mods = 'CTRL|SHIFT', action = act.MoveTabRelative(1) },
{ key = 'LeftArrow',mods = 'CTRL|SHIFT', action = act.ActivateTabRelative(-1) },
{ key = 'RightArrow',mods = 'CTRL|SHIFT', action = act.ActivateTabRelative(1) },
{ key = "_", mods = "CTRL|SHIFT", action=wezterm.action.SplitVertical({domain="CurrentPaneDomain"})},
{ key = "|",mods = "CTRL|SHIFT", action=wezterm.action{SplitHorizontal={domain="CurrentPaneDomain"}}},
{ key = "n", mods = "CTRL|SHIFT", action=wezterm.action{SpawnTab="CurrentPaneDomain"}},
{ key = "z", mods = "CTRL|SHIFT", action="TogglePaneZoomState" },
{ key = "Backspace", mods = "CTRL|SHIFT", action=wezterm.action{CloseCurrentPane={confirm=true}}},
{ key = "Delete", mods = "CTRL|SHIFT", action=wezterm.action{CloseCurrentPane={confirm=false}}},
{ key = "LeftArrow", mods = "CTRL", action=wezterm.action{ActivatePaneDirection="Left"}},
{ key = "DownArrow", mods = "CTRL", action=wezterm.action{ActivatePaneDirection="Down"}},
{ key = "UpArrow", mods = "CTRL", action=wezterm.action{ActivatePaneDirection="Up"}},
{ key = "RightArrow", mods = "CTRL", action=wezterm.action{ActivatePaneDirection="Right"}},

```


| 快捷键   | 应用      | 功能     | 备注  |
| ----- | ------- | ------ | --- |
| M-a   | wezterm | Leader |     |
| C-S-c | wezterm | 复制     |     |
| C-S-v | wezterm | 粘贴     |     |
|       |         |        |     |



### shell级


| 快捷键     | 应用   | 功能                    | 备注  |
| ------- | ---- | --------------------- | --- |
| C-t     | fzf  | fzf查找文件               |     |
| C-r     | fzf  | fzf查找历史命令             |     |
| M-c     | fzf  | fzf查找文件夹              |     |
| M-b/M-f | bash | 前后跳单词                 |     |
| C-W     | bash | kill word (backward)  |     |
| Alt-D   | bash | kill word (forward)   |     |
| C-U     | bash | kill to start of line |     |
| C-K     | bash | kill to end of line   |     |
|         |      |                       |     |


### 应用级(neovim)

设置原则：
1. 需要快速打开的，又不容易产生严重后果的，采用少按键的。
2. 打开后会做很多操作的，或是误触产生严重后果的，采用多按键的。

规则：
全局用Meta和Meta+Shift控制。

| 快捷键 | 应用     | 功能                          | 备注  |
| --- | ------ | --------------------------- | --- |
| M-p | neovim | 工程面板：NerdTree               |     |
| M-P | neovim | 查找Project：Telescope project |     |
| M-l | neovim | 符号列表：Tagbar                 |     |
| M-L | neovim | 搜索符号：Fzflua btags           |     |
| M-f | neovim | 搜索文件：Telescope find_files   |     |
| M-F | neovim | 搜索字符串：Telescope live_grep   |     |
| M-b | neovim | 搜索bufers                    |     |
| M-h | neovim | 搜索help_tags                 |     |
| M-g | neovim | 搜索git_files                 |     |
| M-t | neovim | 打开终端面板: repl                |     |


## Linux command line tools


| name | description       | usage                                                        |
| ---- | ----------------- | ------------------------------------------------------------ |
| jp2a | jpeg to ascii-art | ```bash<br>sudo apt install jp2a<br>jp2a filename.jpg<br>``` |

## 散热能力

空气的导热系数：5-25 ${W}/{m^2K}$
这里的K和摄氏度的单位是相同的。即每高一度，每增加一平方米散热面积，可以散热5~25 J/s 即5~25 W。

手机散热能力是70mA每度，即说每升高一度，可以散度70mA, 4V，即0.28W。
假设空气导热系数是5$W/m^2K$，则手机表面积为 0.28/5 $m^2$ ， 即0.056 $m^2$ ，即5.6平方分米。即 2.36分米见方。显然我们的手机没有这么大，前后面都算上，可能就1.5分米见方，约2.25平方分米。所以这里的空气导热系是按10 $W/m^2K$ 算的？

The difference between `ACaptureSessionOutput` and `ACameraOutputTarget` is fundamental to how the Android NDK Camera2 API structures its session and request hierarchy. Here's a clear breakdown:


## C++ vs Rust

C++和Rust的本质区别是对灵活度的约束。
C++不约束开发者的灵活度。Rust则很大程度上约束开发者的灵活度。
C++不现限开发者的灵活度，这让急于prototyping的开发者很开心，敢让遇到问题摸不着头脑的开发者很沮丧。
其实，给人以灵活度从来都是容易的事情，定义一个约束程度才是真正的难，因为这肯定不会让所有人、所有时间都开心。
很多新的编程语言都开始限制人的自由度，比如说golang，打动我就是它有一个官方的格式化标准。
作为一个做过很多prototyping的人，我觉得还是需要将功夫放在平时，尽量不要欠技术债，欠债总是会要求你还得更多，而且不会让你欠很久。