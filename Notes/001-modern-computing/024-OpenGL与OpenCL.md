
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



# nVidia显卡

## RTX4090

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
