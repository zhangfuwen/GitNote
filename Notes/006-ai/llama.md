---
title: llama.cpp运行Llama的模型的各种方法
---

# PC上运行

## 程序编译

```bash

sudo apt install git ninja-build cmake

git clone https://github.com/ggerganov/llama.cpp

cd llama.cpp
cmake -B build -G Ninja
ninja -C build
cd -
```

## 模型权重文件获取

获取llama模型权重文件：

```bash
if ! command -v pip3; then
  sudo apt install python3 python3-pip
if
pip install transformers # 这个是pyllama的运行时依赖，pyllama可能没写。不安装这个会造成运行时失败
pip install pyllama -U
python -m llama.download --model_size 7B

```
model_size可选值：7B,13B,30B,65B。其中7B模型大概14GB以内，13B模型26GB左右，依些类推，乘以2就行了。
运行时需要把模型完全加载到内存或显存才能工作。
可以通过量化的方式把模型降到原来的四分之一大小。

下载完的目录结构：

```
✅ pyllama_data/tokenizer.model
✅ pyllama_data/tokenizer_checklist.chk
✅ pyllama_data/7B/consolidated.00.pth
✅ pyllama_data/7B/params.json
✅ pyllama_data/7B/checklist.chk
```

模型权重文件是有格式的，这种方式下载到的是pytorch格式的，需要传换成ggml格式才能用：

## 格式转换

```bash
python3 ./llama.cpp/convert-pth-to-ggml.py ./pyllama_data/7B 1
```

## 量化

成生的文件是16位浮点数的，整体size比较大，可以量化为int4的：

```bash
./llamma.cpp/build/bin/quantize ./pyllama_data/7B/ggml-model-f16.bin ./pyllama_data/7B/ggml-model-q4_0.bin 2
```

## 运行

```bash
./llama.cpp/build/bin/main -m ./pyllama_data/7B/ggml-model-q4_0.bin -t 4 -n 128 --color -i -r "User:"
```

# 扩展的中文友好的模型

## Linly

这个git repo上分享是一个tencent pretrain格式的模型：
https://huggingface.co/Linly-AI/ChatFlow-7B

转换为llama格式后可以直接用。

### 下载

```bash
git lfs install
git clone https://huggingface.co/Linly-AI/ChatFlow-7B
git clone https://github.com/ProjectD-AI/llama_inference
```

### 格式转换

从tencent pretrain转换为pytorch格式：

```bash
https://github.com/Tencent/TencentPretrain

mv pyllama_data/7B/consolidated.00.pth{,.back-original} # 备分，将新生成的文件放到pyllama文件夹下

python3 TencentPretrain/scripts/convert_tencentpretrain_to_llama.py --input_model_path ChatFlow-7B/chatflow_7b.bin \
                                                    --output_model_path pyllama_data/7B/consolidated.00.pth \
                                                    --layers 32

```

后续使用需要重新经历[格式转换](#格式转换)、[量化](#量化)、[运行](#运行) 三个步骤。

# 其他运行方式

## llama.cpp其他方式运行

llama.cpp可以使用CLBlast或cuBlas做为后端运行。使用cuBlas可以直接运行在GPU上。

cuBlas:

```bash

cd llama.cpp
rm -rf build
cmake -B build -G Ninja -DLLAMA_CUBLAS=ON
ninja -C build
cd -

```

clBlast:

```bash

cd llama.cpp
rm -rf build
cmake -B build -G Ninja -DLLAMA_CLBLAST=ON # -DCLBlast_dir=/some/path
ninja -C build
cd -

```

运行时需要加上`--gpu-layers`参数。

opencl的话，可以用环境变量指定设备和平台

```bash

GGML_OPENCL_PLATFORM=1 ./main ...
GGML_OPENCL_DEVICE=2 ./main ...
GGML_OPENCL_PLATFORM=Intel ./main ...
GGML_OPENCL_PLATFORM=AMD GGML_OPENCL_DEVICE=1 ./main ...
```

## NDK 编译运行

```bash
cmake -B build -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=29 -DCMAKE_TOOLCHAIN_FILE=/home/xxx/android-ndk-r25c/build/cmake/android.toolchain.cmake  -G Ninja
ninja -C build

```

# conda environment


## cuda build and execute

install cuda tool kit for compilation.

```bash
conda create -n myenv
conda install -c nvidia cuda cuda-nvcc
```

example code(test.cu):

```cuda
#include <stdio.h>

// CUDA kernel to perform vector addition
__global__ void vectorAdd(int* a, int* b, int* c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int size = 1024; // Size of the vectors
    int a[size], b[size], c[size]; // Input and output arrays

    // Initialize input arrays
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    // Declare GPU memory pointers
    int *dev_a, *dev_b, *dev_c;
    // Allocate GPU memory
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_c, size * sizeof(int));

    // Copy input arrays from host to GPU memory
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Launch CUDA kernel on the GPU
    vectorAdd<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, size);

    // Copy the result back from GPU to host memory
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < size; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Free GPU memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}


```

## build and run

```bash
nvcc test.cu -o example
./example
```

# conda install pytorch and pytorch-cuda


## nvidia version matching

nvidia kernel version, toolkit version must roughly match.

to see kernel version, use `nvidia-smi` command:

```bash
nvidia-smi     
Tue Apr 30 18:06:30 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.171.04             Driver Version: 535.171.04   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1080        Off | 00000000:01:00.0  On |                  N/A |
| 32%   45C    P8              12W / 180W |    289MiB /  8192MiB |      9%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

```

` CUDA Version: 12.2  ` is what we want to know.

to figure out cudatoolkit version, use `nvcc --version` command:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0

```

`cuda_12.4.r12.4` is what we want to know.


## install pytorch 

With `pytorch` installed, you can run your model on `cpu`.
With `pytorch-cuda` installed, you can run your model on `cuda`.

Install pytorch:

```bash 
conda install pytorch=2.2
```
or

```bash
conda install pytorch==2.2.2
```

Single equal sign indicates none exact match, while double equal sign means exact match.



To test if cuda is available:

```
python3
>>> import torch 
>>> print(torch.cuda.is_available())
False
```

Install pytorch-cuda:

```bash
conda install pytorch-cuda=12.4 -c pytorch
```

The version better matches the cuda version you got using `nvidia-smi` command.

Sometimes, the version you wanted does not exist, try some other versions as long as the major version matches:

```
conda search pytorch-cuda -c pytorch
Loading channels: done
# Name                       Version           Build  Channel             
pytorch-cuda                    11.6      h867d48c_0  pytorch             
pytorch-cuda                    11.6      h867d48c_1  pytorch             
pytorch-cuda                    11.7      h67b0de4_0  pytorch             
pytorch-cuda                    11.7      h67b0de4_1  pytorch             
pytorch-cuda                    11.7      h67b0de4_2  pytorch             
pytorch-cuda                    11.7      h778d358_3  pytorch             
pytorch-cuda                    11.7      h778d358_5  pytorch             
pytorch-cuda                    11.8      h7e8668a_3  pytorch             
pytorch-cuda                    11.8      h7e8668a_5  pytorch             
pytorch-cuda                    11.8      h8dd9ede_2  pytorch             
pytorch-cuda                    12.1      ha16c6d3_5  pytorch             
pytorch-cuda                    12.4      hc786d27_6  pytorch 
```

For example, if your cuda version is 11.3, you may try 11.7 which has backward compatibility with version 11.3.

Sometimes, pytorch-cuda would have dependency conflicts with python version. so you may downgrade to an older version of python as well:

```bash
conda install python=3.8 pytorch-cuda=12.4  -c pytorch -v
```

better follow instructions here: https://pytorch.org/get-started/previous-versions/
