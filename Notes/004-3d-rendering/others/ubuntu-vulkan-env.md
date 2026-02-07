

# ubuntu下vulkan开发环境搭建


## 1. OpenGL 环境搭建

```bash
sudo apt install libglfw3-dev git vim mesa-utils

git clone https://github.com/erf/triangle
```

下载glad: https://glad.dav1d.de/generated/tmpC0LWOyglad/

编译:
```bash
g++ -g -I../glad/include main.c ../glad/src/glad.c -lglfw -ldl
```

注意,：glad初始化要放在glfw初始化上下文之后。

## 2. vulkan环境搭建

