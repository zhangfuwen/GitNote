
# OpenGL

下面一一段简单的OpenCL代码，用于绘制一个三角形：

```c

1 const char *vertexShaderSource = "#version 330 core\n"
  2 "layout (location = 0) in vec3 aPos;// 位置变量的属性位置值为 0 \n"
  3 "void main()\n"
  4 "{\n"
  5 "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
  6 "}\0";
  7 
  8 const char *fragmentShaderSource = "#version 330 core\n"
  9 "out vec4 FragColor;\n"
 10 "void main()\n"
 11 "{\n"
 12 "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);//最终的输出颜色\n"
 13 "}\0";
 14 
 15 int hello_triangle()
 16 {
 17     GLFWwindow* window = init_window();
 18 
 19     ///定义着色器
 20     //创建一个顶点着色器对象，注意还是用ID来引用的
 21     unsigned int vertexShader;
 22     vertexShader = glCreateShader(GL_VERTEX_SHADER);
 23 
 24     //着色器源码附加到着色器对象上
 25     glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);//要编译的着色器对象作为第一个参数。第二参数指定了传递的源码字符串数量，这里只有一个。第三个参数是顶点着色器真正的源码，第四个参数我们先设置为NULL
 26     glCompileShader(vertexShader);//编译源码
 27     int  success;
 28     char infoLog[512];
 29     glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);//用glGetShaderiv检查是否编译成功
 30     if (!success)
 31     {
 32         glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
 33         std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
 34     }
 35 
 36     //创建一个片段着色器对象，注意还是用ID来引用的
 37     unsigned int fragmentShader;
 38     fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
 39     glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
 40     glCompileShader(fragmentShader);//编译源码
 41     glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);//用glGetShaderiv检查是否编译成功
 42     if (!success)
 43     {
 44         glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
 45         std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
 46     }
 47 
 48     //创建一个着色器对程序
 49     unsigned int shaderProgram;
 50     shaderProgram = glCreateProgram();
 51     glAttachShader(shaderProgram, vertexShader);//把之前编译的着色器附加到程序对象上
 52     glAttachShader(shaderProgram, fragmentShader);
 53     glLinkProgram(shaderProgram);//glLinkProgram链接它们
 54     glGetProgramiv(shaderProgram, GL_COMPILE_STATUS, &success);//用glGetProgramiv检查是否编译成功
 55     if (!success)
 56     {
 57         glGetShaderInfoLog(shaderProgram, 512, NULL, infoLog);
 58         std::cout << "ERROR::SHADER::PROGRAM::LINK_FAILED\n" << infoLog << std::endl;
 59     }
 60 
 61     //链接后即可删除
 62     glDeleteShader(vertexShader);
 63     glDeleteShader(fragmentShader);//*/
 64 
 65     ///定义顶点对象
 66     float vertices[] = {
 67     -0.5f, -0.5f, 0.0f,
 68     0.5f,-0.5f, 0.0f,
 69     0.0f, 0.5f, 0.0f  70     
 71     };
 72 
 73     //生成VAO对象，缓冲ID为VAO
 74     unsigned int VAO;
 75     glGenVertexArrays(1, &VAO);
 76     glBindVertexArray(VAO);//绑定VAO,从绑定之后起，我们应该绑定和配置对应的VBO和属性指针，之后解绑VAO，供之后使用
 77 
 78     //生成VBO对象，缓冲ID为VBO
 79     unsigned int VBO;
 80     glGenBuffers(1, &VBO);//第一个参数GLsizei是要生成的缓冲对象的数量，第二个GLuint是要输入用来存储缓冲对象名称的数组
 81 
 82     //绑定到目标对象,VBO变成了一个顶点缓冲类型
 83     glBindBuffer(GL_ARRAY_BUFFER, VBO);//第一个就是缓冲对象的类型，第二个参数就是要绑定的缓冲对象的名称
 84     glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);//数据传入缓冲内存中,GL_STATIC_DRAW：数据不会或几乎不会改变； GL_DYNAMIC_DRAW：数据会被改变很多； GL_DYNAMIC_DRAW：数据会被改变很多
 85 
 86     //设置顶点属性指针，如何解析顶点数据
 87     /*
 88     第一个参数指定我们要配置的顶点属性，顶点着色器中使用layout(location = 0)定义
 89     第二个参数指定顶点属性的大小
 90     第三个参数指定数据的类型
 91     第四个参数定义我们是否希望数据被标准化(Normalize)。如果我们设置为GL_TRUE，所有数据都会被映射到0（对于有符号型signed数据是-1）到1之间
 92     第五个参数步长(Stride)，它告诉我们在连续的顶点属性组之间的间隔
 93     最后一个参数的类型是void*，数据在缓冲中起始位置的偏移量(Offset)
 94     */
 95     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
 96     glEnableVertexAttribArray(0);//启用顶点属性layout(location = 0)，顶点属性默认是禁用的
 97     glBindBuffer(GL_ARRAY_BUFFER, 0);//设置完属性，解绑VBO
 98 
 99     glBindVertexArray(0);//配置完VBO及其属性，解绑VAO
100 
101 
102     //绘制模式为线条GL_LINE，填充面GL_FILL
103     //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);//正反面
104 
105     while (!glfwWindowShouldClose(window))
106     {
107         processInput(window);
108 
109 
110         //清空屏幕
111         glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
112         glClear(GL_COLOR_BUFFER_BIT);
113 
114 
115         //绘制物体
116         glUseProgram(shaderProgram);//激活程序对象
117 
118         glBindVertexArray(VAO);
119         //使用VAO绘制
120         glDrawArrays(GL_TRIANGLES, 0, 3);//绘制图元为三角形，起始索引0，绘制顶点数量3
121 
122         glfwSwapBuffers(window);//交换颜色缓冲（它是一个储存着GLFW窗口每一个像素颜色值的大缓冲）
123         glfwPollEvents();//检查有没有触发什么事件
124     }
125 
126     //释放对象
127     glDeleteVertexArrays(1, &VAO);
128     glDeleteBuffers(1, &VBO);
129 
130     std::cout << "finish!" << std::endl;
131     glfwTerminate();//释放/删除之前的分配的所有资源
132     return 0;
133 }
```

代码很长，但功能很简单。它在120行每次调用glDrawArrays的时候，调用一次渲染流水线。流水线（管线）会执行一些固定的部分，也会执行两个shader，即vertex shader和fragment shader。
1. 流水线在固定的部分中将输入顶点进行一定的加工，之后会传给vertex shader，vertex shader对单个顶点进行处理，输出一个处理后的顶点。在上述代码中，vertex shader几乎没有做什么事情，它只是把每个顶点的w分量设置为1.0.这代码顶点的部分没有进行任何的变换。你也可以修改代码，给每个顶点的位置加上一些偏移，或者缩放，你会看到最终输出的图像可能就不是一个三角形了。
2. 在vertex shader之后，流水线的固定部分会将这些顶点组装成三角形，然后栅格化，根据viewport的大小，决定每个三角形中取多少个点。然后将这些点送给fragment shader。上述代码中，fragment shader也几乎没做什么事情，它只是给每个点一个固定的颜色。这样最终显示的三角形里面就被填充为这种颜色。

当技术发展到这一部分的时候，人们就发现，vertex shader和fragment shader具有一些特点：
1. 虽然要对很多点或像素（片元）进行计算，但是所做的计算的公式都是相同的，只有输入输出的数据不同。
2. vertex shader和fragment shader可以复用一部分硬件逻辑单元。

现实的计算任务中，有些任务（特别是大量数据情况下的任务）也具有计算步骤相同但数据不同的特点。所以也可以用显卡的这个单元来做这个计算。

OpenGL ES 3.0版本的时候就引入了compute shader的概念，只用来做通用计算。

compute shader的用法大概这样：

```c
char * computeShaderSource = "#version 310 es 
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in; 
layout (std430, binding = 0) buffer DataBuffer { 
	float data[]; 
} buffer1; 
void main() { 
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy); 
	buffer1.data[pos.y * int(gl_NumWorkGroups.x) + pos.x] *= float(pos.y); 
}
"

GLuint GLUtils::LoadComputeShader(const char* computeShaderSource) { 
	GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER); 
	glShaderSource(computeShader, 1, &computeShaderSource, NULL); 
	glCompileShader(computeShader); 
	GLint success; 
	glGetShaderiv(computeShader, GL_COMPILE_STATUS, &success); 
	if (!success) { 
		GLchar infoLog[512]; 
		glGetShaderInfoLog(computeShader, 512, NULL, infoLog); 
		LOGCATE("GLUtils::LoadComputeShader Compute shader compilation failed: %s", infoLog); 
		return 0; 
	} 
	GLuint computeProgram = glCreateProgram(); 
	glAttachShader(computeProgram, computeShader); 
	glLinkProgram(computeProgram); 
	glGetProgramiv(computeProgram, GL_LINK_STATUS, &success); 
	if (!success) { 
		GLchar infoLog[512]; 
		glGetProgramInfoLog(computeProgram, 512, NULL, infoLog); 
		LOGCATE("GLUtils::LoadComputeShader Compute shader linking failed: %s", infoLog); 
		return 0; 
	} 
	glDeleteShader(computeShader); 
	return computeProgram; 
}

glGenBuffers(1, &m_DataBuffer); 
glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_DataBuffer); 
float data[2][4] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}; 
glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data), data, GL_DYNAMIC_COPY); 
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_DataBuffer);//binding=0
// Use the program object glUseProgram (m_ProgramObj); //2x4 int numGroupX = 2; int numGroupY = 4;
glDispatchCompute(numGroupX, numGroupY, 1); 
glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT); // 读取并打印处理后的数据 
glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_DataBuffer); 
auto* mappedData = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, m_DataSize, GL_MAP_READ_BIT); LOGCATE("ComputeShaderSample::Draw() Data after compute shader:\n"); 
for (int i = 0; i < m_DataSize/ sizeof(float); ++i) { 
	LOGCATE("ComputeShaderSample::Draw() => %f", mappedData[i]); 
} 
glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

```

这段代码中的compute shader直接从storage buffer读取输入，再将输出写入到storage buffer。


# OpenCL


OpenCL是一个专门于通用计算任务的API。目前还比较尴尬。理论上来讲，它可以应用于各种各样的硬件，但实际情况是除了GPU外，支持OpenCL的硬件并不很多，使用OpenCL的开发者也比较少。

它的API接口类似于OpenGL，但是提供更精细的command queue管理，这点有点类似于vulkan。

由于开发者和硬件vendor的双重不待见，它的性能如何其实是打个问号的。所以一般是万不得已才会用。

详细可以参考：https://zhuanlan.zhihu.com/p/602844623

# nVidia显卡

## 显卡与cuda发展历程

### 显卡发展历程

cuda是2006年提出的。

以下是NVIDIA显卡几个重要版本的重要时间节点：

1. GeForce 256
	- **发布时间**：1999年8月。这是NVIDIA正式发布的第一个GeForce产品，它正式定义了GPU词汇，并且支持硬件T&L引擎，采用0.22微米制程打造，拥有四条渲染流水线，达到2300万个晶体管，分为SDRAM和DDR SDRAM两种版本.
2. GeForce 2系列
	- **发布时间**：2000年至2001年。这一代产品分为面向主流的GeForce 2 MX系列、中高端的GeForce 2（GTS、Pro、Ti、Ultra）系列以及全球首款独立移动图形处理器GeForce 2 GO系列，形成了较为完整的产品线.
3. GeForce 3系列
	- **发布时间**：2001年。这是第三代GeForce产品，率先支持DirectX 8和可编程T&L引擎，主要分为GeForce 3、GeForce 3 Ti 200和GeForce 3 Ti 500.
4. GeForce 4系列
	- **发布时间**：2002年。实际上是在第三代基础架构上进行改良，增加额外的顶点着色引擎，支持Accuview高效反锯齿技术，更高的核心和显存频率，显示存储器控制器有算法改进，桌面端主要分为GeForce 4 Ti系列和GeForce 4 MX系列.
5. GeForce FX系列
- **发布时间**：2002年至2004年。全系列完全支持DirectX 9.0标准，其中FX 5800和FX 5800 Ultra还采用了三星刚量产的GDDR2显存。此外，还出现了率先支持PCI-E的多款型号，即PCX系列.
6. GeForce 6系列
	- **发布时间**：2004年至2006年。率先亮相的最高端的GeForce 6800 Ultra带来了16条渲染流水线和6组顶点着色的质变规格，同时采用了当时最新的GDDR3显存，频率突破轻松1GHz，性能相比上代FX旗舰显卡增长两倍以上。此外，还增加支持Shader Model 3.0着色、Intellisample 3.0抗锯齿、PureVideo视频处理等技术，并重新优化推出了SLI技术.
7. GeForce 7系列
	- **发布时间**：2005年至2006年。全系列原生支持PCI-E，AGP从此完全淘汰。首当其冲的GeForce 7800（G70）系列则是使用更为成熟的0.11微米制程工艺，到了中后期阶段的型号才使用80、90纳米制程工艺.
8. GeForce 8系列
	- **发布时间**：2006年11月。这一系列显卡采用了全新的统一渲染架构，在性能和功耗方面都有了显著的提升，标志着NVIDIA显卡进入了一个新的发展阶段 。**CUDA** 最初是针对 NVIDIA 的 GeForce 8 系列显卡设计的 。GeForce 8 系列采用了全新的统一渲染架构，为 CUDA 的实现提供了良好的硬件基础。这种架构使得 GPU 的处理单元能够更加灵活地分配和执行不同类型的计算任务，有利于 CUDA 充分发挥 GPU 的并行计算能力，从而实现高性能的通用计算 。

	以下是2006年到2018年NVIDIA发布的部分主要显卡及其特点：
	
	
	
		### GeForce 8系列（2007年）
		- **GeForce 8800 GTX**：采用了全新的统一渲染架构，是显卡架构的一次重大变革。它拥有众多的流处理器，能够提供强大的图形处理能力，在DirectX 10的应用程序中表现出色，为当时的高端游戏和图形应用提供了顶级的性能支持，巩固了NVIDIA在高端显卡市场的地位.
		
		### GeForce 9系列（2008年）
		- **GeForce 9600 GT**：基于GeForce 8系列的架构进行了改进，采用65nm工艺制程，流处理器削减到64个，功耗相对降低，但性能、功耗比表现出色，性价比相比GeForce 8800 GT更高，在中高端市场具有较强的竞争力.
		
		### GeForce GTX 200系列（2009年）
		- **GeForce GTX 280**：采用65nm工艺制程，拥有240个流处理器，支持双精度浮点运算。该系列引入了CUDA基础架构，得以支持新的物理引擎，开启了NVIDIA显卡在通用计算领域的新篇章，为后来的深度学习等应用奠定了基础.
		
		### GeForce GTX 400系列（2010年）
		- **GeForce GTX 480**：采用Fermi架构，完整的GF100核心带来了强大的性能，但同时也伴随着高功耗与发热量的问题，不过其在图形处理能力和计算性能方面相比前代有了显著提升，能够更好地应对日益复杂的游戏画面和计算任务.
		
		### GeForce GTX 500系列（2011年）
		- **GeForce GTX 580**：采用GF110核心，拥有完整的512个流处理器，相比上一代GTX 480，在纹理采样与算法上做了优化，性能得到进一步提升，能够流畅运行当时的各类大型3D游戏.
		- **GeForce GTX 560 Ti**：采用全新的GF114架构，从CUDA核心和SM单元做了优化调整，性能与价格平衡得较好，在中高端显卡市场中获得了不错的响应，为中高端用户提供了性价比之选.
		
		### GeForce GTX 600系列（2012年）
		- **GeForce GTX 680**：采用Kepler架构，相比前代在性能和功耗控制上有了进一步优化，能够在较低的功耗下提供出色的图形性能，其出色的性能和相对合理的功耗，使其成为当时中高端市场的热门选择.
		- **GeForce GTX 690**：双芯显卡，拥有两颗强大的GPU核心，性能极其强劲，能够满足最苛刻的游戏玩家和专业图形用户的需求，但价格也相对较高，定位为顶级高端市场.
		
		### GeForce GTX 700系列（2013年）
		- **GeForce GTX 780**：基于Kepler架构的进一步优化，性能相比前代有所提升，能够为游戏玩家带来更流畅的游戏体验和更细腻的画面效果。同时，在功耗和散热方面也有所改进，提高了显卡的稳定性和可靠性 。
		- **GeForce GTX 780 Ti**：该系列的旗舰型号，拥有更多的CUDA核心和更高的频率，性能更加强劲，能够在高分辨率和高画质下流畅运行各种大型3D游戏，是当时追求极致性能的游戏玩家和专业图形用户的首选。
		
		### GeForce GTX 900系列（2014年）
		- **GeForce GTX 970**：采用Maxwell架构，拥有较高的性能和较低的功耗，在性能和功耗比方面表现出色。其出色的能耗比使得它在主流市场中广受欢迎，能够满足大多数用户对于游戏和日常图形应用的需求。
		- **GeForce GTX 980**：作为该系列的高端型号，性能更加强劲，拥有更多的CUDA核心和更高的显存带宽，能够在高分辨率下提供流畅的游戏体验和出色的图形性能，为高端游戏玩家和专业图形用户提供了强大的支持。
		
		### GeForce GTX 10系列（2016年）
		- **GeForce GTX 1060**：采用Pascal架构，具有较高的性价比，能够在1080p分辨率下流畅运行大多数游戏，是主流游戏玩家的热门选择。其性能与上一代相比有了显著提升，同时功耗控制也较为出色。
		- **GeForce GTX 1070**：性能更加强劲，能够在高分辨率和高画质下流畅运行各类大型3D游戏，与同级别竞争对手相比具有一定的性能优势，为中高端游戏玩家提供了出色的游戏体验。
		- **GeForce GTX 1080**：该系列的高端型号，采用16nm FinFET工艺，拥有大量的CUDA核心和高速的GDDR5X显存，性能极其出色，能够满足4K分辨率下的游戏需求，是当时追求极致游戏性能的玩家的首选显卡之一 。
		
		### GeForce GTX 11系列（2017年）
		- **GeForce GTX 1160/1170等**：这一系列在性能和功耗上进行了进一步优化，相较于前代产品，在相同功耗下能够提供更出色的性能表现，为玩家带来了更好的游戏体验，在主流和中高端市场中都具有一定的竞争力，不过该系列并非重大架构变革的产品，而是对Pascal架构的优化和升级。
		
		### GeForce GTX 12系列（2018年）
		- **GeForce GTX 1260等**：继续在性能和功耗方面进行平衡和优化，以适应不同用户群体的需求，性能相比前代有所提升，能够满足各类游戏和图形应用的需求，同时在价格方面也具有一定的合理性，为主流用户提供了可靠的选择。需要注意的是，GTX 12系列同样是在已有架构基础上的改进和优化，并非全新架构的产品 。
   
9. GeForce 20系列
	- **发布时间**：2018年8月至9月。 这一系列显卡引入了光线追踪技术，为游戏玩家带来了更加逼真的光影效果，代表型号有GeForce RTX 2080、GeForce RTX 2070等 。
10. GeForce 30系列
	- **发布时间**：2020年9月至10月 。相比上一代产品，在性能和光线追踪能力上都有了进一步的提升，例如GeForce RTX 3090、GeForce RTX 3080等型号，受到了广大游戏玩家和专业创作者的关注 。
11. GeForce 40系列
	- **发布时间**：2022年至2023年 。这一系列继续优化了性能和光线追踪技术，同时在功耗控制和散热设计上也有所改进，像GeForce RTX 4090、GeForce RTX 4080等型号，为用户提供了更强大的图形处理能力 。

以下是部分NVIDIA显卡的CUDA Core和Tensor Core数量：

| 显卡系列 | CUDA Core数量 | Tensor Core数量 |
| ---- | ---- | ---- |
| GeForce 7系列 | 无 | 无 |
| GeForce 8系列 | 无 | 无 |
| GeForce 9系列 <br> GTX 970 | 1664 | 无 |
| GTX 980 | 2048 | 无 |
| GTX 980 Ti | 2816 | 无 |
| GeForce GTX 10系列 <br> GTX 1060 6GB | 1280 | 无 |
| GTX 1070 | 1920 | 无 |
| GTX 1080 | 2560 | 无 |
| GeForce GTX 11系列 <br> GTX 1160Ti | 1536 | 无 |
| GTX 1170 | 2048 | 无 |
| GeForce RTX 20系列 <br> RTX 2060 | 1920 | 30 |
| RTX 2070 | 2304 | 40 |
| RTX 2080 | 2944 | 68 |
| RTX 2080 Ti | 4352 | 68 |
| GeForce RTX 30系列 <br> RTX 3060 | 3584 | 28 |
| RTX 3070 | 5888 | 46 |
| RTX 3080 | 8704 | 68 |
| RTX 3090 | 10496 | 82 |
| GeForce RTX 40系列 <br> RTX 4060 | 3072 | 34 |
| RTX 4070 | 5888 | 60 |
| RTX 4080 | 7680 | 96 |
| RTX 4090 | 16384 | 128 | 

### cuda core

GeForce 8系列没有cuda core，但当时的 GPU 架构已经具备了一定的并行处理能力基础，为 CUDA 的支持提供了可能。当时 GPU 的可编程性逐渐增强，有了如可编程着色器等技术，这些为后来 CUDA 的引入和发展奠定了架构基础，使得开发者能够逐渐挖掘 GPU 在通用计算方面的潜力。

SIMT（Single Instruction, Multiple Threads）即单指令多线程，是 NVIDIA GPU 架构中的一种执行模型。在这种模型下，多个线程在同一时刻执行同一条指令，但操作的数据可以不同，从而实现数据并行性。

在 SIMT 架构中，线程被组织成线程束（warp），一个线程束通常包含 32 个线程。当执行一个指令时，这 32 个线程会同时执行相同的操作，但作用于不同的数据元素。例如在进行向量加法运算时，一个线程束中的 32 个线程会同时对各自对应的向量元素进行加法操作。

与 CUDA Core 的关系：CUDA Core 的工作方式基于 SIMT 模型。多个 CUDA Core 组成一个流多处理器（SM），在 SM 中，线程束被分配到不同的 CUDA Core 上执行。CUDA Core 利用 SIMT 的并行性，在每个时钟周期内对多个线程执行相同的指令，实现高效的并行计算.

SIMT 模型充分发挥了 GPU 的并行计算能力，通过大量线程的并行执行，大大提高了计算效率，尤其适用于处理大规模数据的并行计算任务，如深度学习中的神经网络训练和推理、图形渲染等场景，能够在短时间内处理海量的数据。### tensor core


传统的 CUDA Core 虽然能够为通用计算提供并行加速，但在处理深度学习中频繁出现的大规模矩阵乘法和卷积运算时，效率逐渐难以满足需求。因为每个 CUDA Core 在一个时钟周期内只能执行一个操作，对于复杂的矩阵运算，单纯依靠增加 CUDA Core 的数量和提高时钟频率已经难以实现性能的大幅提升.



Tensor Core通过以下方式解决了传统CUDA Core在处理大规模矩阵乘法和卷积运算时效率不足的问题：
1. 采用混合精度计算
	- **降低精度提升速度**：Tensor Core支持混合精度计算，通常在计算过程中使用半精度（FP16）或其他低精度数据类型，而在输入输出时使用单精度（FP32）或更高精度以保证结果准确性。相比传统CUDA Core使用的全精度计算，FP16数据类型的位宽更小，数据处理速度更快，从而能够在相同时间内处理更多的数据，大幅提高计算吞吐量.
	- **精度调整与权衡**：可以根据应用场景灵活调整计算精度，用户可在性能和精度之间做出权衡。对于一些对精度要求不是极其严格的场景，通过降低精度来进一步提升计算效率，满足不同应用对精度和速度的不同需求.
2. 优化矩阵乘法执行方式
	- **并行处理小矩阵乘法**：专门针对深度学习中常见的小矩阵乘法运算进行了优化，如Volta架构中的Tensor Core可同时处理一个4x4x4的张量运算，Ampere架构下的Tensor Core能够处理16x16x16的FP16或TF32矩阵乘法。通过高度并行化的矩阵乘法和累加操作，在一个时钟周期内完成大量这类小矩阵乘法运算，显著提高了矩阵乘法的执行效率.
	- **硬件级别的优化**：从硬件架构层面进行优化，减少计算和存储带宽需求。例如采用特殊的电路设计和数据通路，能够更高效地处理矩阵乘法中的数据流动，降低数据传输瓶颈，节省高带宽内存需求，从而提升整个系统的性能表现.
3. 具备强大的并行处理能力
	- **利用GPU并行架构**：借助GPU本身的并行处理架构优势，Tensor Core能够同时执行海量的矩阵运算，为大型神经网络的训练及推理提供强大的密集计算支持，充分发挥GPU的并行计算潜力，满足深度学习中大规模矩阵乘法和卷积运算对并行计算的高要求.

### Tensor Core与SIMD的区别

传统的SIMD在矩阵运算中有一些局限，比如说两个向量乘，它的需要两个向量对应位置的数相乘再累加。传统SIMD可以一个cycle完成相乘，但累加就不能一步完成，只能采用类似折半查找的方式通过lgN步完成计算，实际开发中也有一些trick来加速，但通用性不强。

Tensor Core不完全等同于传统意义上的SIMD（单指令多数据），但在某些方面具有SIMD的特性，以下是具体分析：

1. 与SIMD的相似之处
	- **并行执行相同操作**：在执行矩阵乘法等运算时，Tensor Core可以在一个时钟周期内对多个数据元素同时执行相同的指令，这与SIMD的基本概念相符，即多个处理单元同时执行同一条指令，但操作不同的数据，从而实现数据级并行，提高计算效率.
	- **数据并行计算加速**：SIMD通过并行处理多个数据元素来加速计算，而Tensor Core也是专门针对深度学习中常见的大规模矩阵乘法和卷积等数据并行度极高的运算进行优化设计，能够在同一时间内处理大量的数据，充分发挥了数据并行计算的优势，加快了计算速度.

2.  与SIMD的不同之处
	- **操作的复杂性和灵活性**：传统SIMD通常针对简单的数据类型和操作，如整数或浮点数的加法、乘法等基本算术运算。而Tensor Core主要用于处理矩阵乘法累加操作以及与深度学习相关的更复杂的计算模式，支持混合精度计算等特殊功能，并且能够根据不同的深度学习模型和任务需求，灵活地配置和调整计算精度及数据类型，其操作的复杂性和灵活性远远超出了传统SIMD的范畴.
	- **硬件架构和实现细节**：Tensor Core是NVIDIA GPU中的特定硬件单元，其内部架构和实现方式与传统SIMD处理器有所不同 。它具有专门为加速矩阵乘法而设计的电路和逻辑，能够高效地处理大规模矩阵数据，并通过优化的数据通路和存储层次结构，实现了更高的计算性能和能效比。此外，Tensor Core还与GPU的其他组件紧密协作，如CUDA Core、缓存、内存控制器等，形成了一个完整的异构计算架构，以满足深度学习应用对计算资源的高要求.
	- **编程模型和指令集**：在编程模型上，虽然CUDA编程中warp内的线程以SIMD方式执行，但Tensor Core有其独特的编程接口和指令集，开发人员需要使用特定的函数和指令来调用Tensor Core进行计算，如NVIDIA提供的cuBLAS、cuDNN等库函数，这些函数内部会自动利用Tensor Core的加速能力。与传统SIMD的编程方式相比，Tensor Core的编程模型更加高级和抽象，开发人员无需直接操作底层的硬件指令，降低了编程的复杂性，但同时也需要对深度学习框架和相关库有一定的了解和掌握.## RTX4090

## A100

## CUDA

# 高通平台

## Snapdragon上的GPU

## Snapdragon上的NPU


# MTK平台

# nVidia的显卡都是显卡吗？

2023前以来的大模型热导致英伟达公司热得发烫。为什么AI要用显卡来做计算呢？
A100显卡的虽然中文叫显卡，英文叫GPGPU，但其实它跟显示或Graphics已经没有太大的关系了。

A100
